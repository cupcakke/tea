import std/[asyncdispatch, asynchttpserver, asyncnet, json, strutils, strformat,
            os, times, tables, sets, sequtils, math, algorithm, random, options,
            locks, hashes, base64, uri, deques, monotimes, sha1, mimetypes,
            httpclient, streams, parseutils, osproc, asyncstreams, net, nativesockets,
            atomics]

let
  DbFile = getEnv("AGENT_DB_PATH", getEnv("AGENT_DB", "agent_runtime.db"))
  ModularBaseUrl = getEnv("MODULAR_BASE_URL", "https://api.modular.com/v1").strip(chars = {'/'})
  ModularModel = getEnv("MODULAR_MODEL", "zai-org/glm-5.3")
  ServerPort = parseInt(getEnv("PORT", "8080"))
  WorkspaceRoot = getEnv("AGENT_WORKSPACE", "workspace")
  KnowledgeRoot = getEnv("AGENT_KNOWLEDGE", "knowledge")

const
  MaxTokens = 100000
  Temperature = 0.96
  TopP = 1.0
  FrequencyPenalty = 0.8
  PresencePenalty = 0.5
  Seed = 1234
  System1HzInterval = 50
  System2HzInterval = 1000
  MaxRetryPerStep = 3
  MaxStepsDefault = 1000
  EmbeddingDim = 128
  RrfK = 60.0
  TokenBudgetDefault = 100_000_000
  MaxStateBytes = 131072
  MaxPromptBytes = 262144
  HttpTimeoutMs = 120000
  MaxHttpBodyBytes = 131072
  SqliteLib = when defined(windows): "sqlite3_64.dll" elif defined(macosx): "libsqlite3.dylib" else: "libsqlite3.so(|.0)"

type
  SqliteDb = ptr object
  SqliteStmt = ptr object

{.push importc, cdecl, dynlib: SqliteLib.}
proc sqlite3_open_v2(filename: cstring, ppDb: ptr SqliteDb, flags: cint, zVfs: cstring): cint
proc sqlite3_close_v2(db: SqliteDb): cint
proc sqlite3_exec(db: SqliteDb, sql: cstring, callback: pointer, arg: pointer, errmsg: ptr cstring): cint
proc sqlite3_prepare_v2(db: SqliteDb, zSql: cstring, nByte: cint, ppStmt: ptr SqliteStmt, pzTail: ptr cstring): cint
proc sqlite3_step(pStmt: SqliteStmt): cint
proc sqlite3_finalize(pStmt: SqliteStmt): cint
proc sqlite3_reset(pStmt: SqliteStmt): cint
proc sqlite3_bind_text(pStmt: SqliteStmt, idx: cint, value: cstring, n: cint, destructor: pointer): cint
proc sqlite3_bind_int64(pStmt: SqliteStmt, idx: cint, value: int64): cint
proc sqlite3_bind_double(pStmt: SqliteStmt, idx: cint, value: float64): cint
proc sqlite3_bind_null(pStmt: SqliteStmt, idx: cint): cint
proc sqlite3_column_count(pStmt: SqliteStmt): cint
proc sqlite3_column_text(pStmt: SqliteStmt, iCol: cint): cstring
proc sqlite3_column_int64(pStmt: SqliteStmt, iCol: cint): int64
proc sqlite3_column_double(pStmt: SqliteStmt, iCol: cint): float64
proc sqlite3_column_type(pStmt: SqliteStmt, iCol: cint): cint
proc sqlite3_column_name(pStmt: SqliteStmt, iCol: cint): cstring
proc sqlite3_errmsg(db: SqliteDb): cstring
proc sqlite3_free(p: pointer)
proc sqlite3_busy_timeout(db: SqliteDb, ms: cint): cint
proc sqlite3_last_insert_rowid(db: SqliteDb): int64
proc sqlite3_changes(db: SqliteDb): cint
{.pop.}

const
  SQLITE_OK = 0.cint
  SQLITE_ROW = 100.cint
  SQLITE_DONE = 101.cint
  SQLITE_NULL = 5.cint
  SQLITE_OPEN_READWRITE = 0x00000002.cint
  SQLITE_OPEN_CREATE = 0x00000004.cint
  SQLITE_OPEN_FULLMUTEX = 0x00010000.cint
  SQLITE_TRANSIENT = cast[pointer](-1)

type
  DbError = object of CatchableError

  Store = ref object
    handle: SqliteDb
    lock: Lock
    path: string

  Row = Table[string, JsonNode]

proc raiseDb(s: Store, ctx: string) =
  let msg = $sqlite3_errmsg(s.handle)
  raise newException(DbError, ctx & ": " & msg)

proc openStore(path: string): Store =
  var db: SqliteDb
  let flags = SQLITE_OPEN_READWRITE or SQLITE_OPEN_CREATE or SQLITE_OPEN_FULLMUTEX
  if sqlite3_open_v2(path.cstring, addr db, flags, nil) != SQLITE_OK:
    raise newException(DbError, "cannot open database at " & path)
  discard sqlite3_busy_timeout(db, 15000.cint)
  result = Store(handle: db, path: path)
  initLock(result.lock)

proc execRawUnlocked(s: Store, sql: string) =
  var err: cstring
  if sqlite3_exec(s.handle, sql.cstring, nil, nil, addr err) != SQLITE_OK:
    var m = "sql error"
    if err != nil:
      m = $err
      sqlite3_free(err)
    raise newException(DbError, m & " :: " & sql)

proc execRaw(s: Store, sql: string) =
  acquire(s.lock)
  defer: release(s.lock)
  execRawUnlocked(s, sql)

proc bindParams(s: Store, st: SqliteStmt, params: seq[JsonNode]) =
  for i, p in params:
    let idx = (i + 1).cint
    case p.kind
    of JNull:
      discard sqlite3_bind_null(st, idx)
    of JInt:
      discard sqlite3_bind_int64(st, idx, p.getBiggestInt())
    of JFloat:
      discard sqlite3_bind_double(st, idx, p.getFloat())
    of JBool:
      discard sqlite3_bind_int64(st, idx, if p.getBool(): 1 else: 0)
    of JString:
      let v = p.getStr()
      discard sqlite3_bind_text(st, idx, v.cstring, v.len.cint, SQLITE_TRANSIENT)
    else:
      let v = $p
      discard sqlite3_bind_text(st, idx, v.cstring, v.len.cint, SQLITE_TRANSIENT)

proc query(s: Store, sql: string, params: seq[JsonNode] = @[]): seq[Row] =
  acquire(s.lock)
  defer: release(s.lock)
  var st: SqliteStmt
  if sqlite3_prepare_v2(s.handle, sql.cstring, -1.cint, addr st, nil) != SQLITE_OK:
    raiseDb(s, "prepare failed for " & sql)
  defer: discard sqlite3_finalize(st)
  bindParams(s, st, params)
  result = @[]
  while true:
    let rc = sqlite3_step(st)
    if rc == SQLITE_ROW:
      var row = initTable[string, JsonNode]()
      let n = sqlite3_column_count(st)
      for c in 0 ..< n:
        let name = $sqlite3_column_name(st, c)
        let ct = sqlite3_column_type(st, c)
        if ct == SQLITE_NULL:
          row[name] = newJNull()
        else:
          let raw = sqlite3_column_text(st, c)
          if raw == nil:
            row[name] = newJNull()
          else:
            row[name] = newJString($raw)
      result.add(row)
    elif rc == SQLITE_DONE:
      break
    else:
      raiseDb(s, "step failed for " & sql)

proc execUnlocked(s: Store, sql: string, params: seq[JsonNode] = @[]): int64 =
  var st: SqliteStmt
  if sqlite3_prepare_v2(s.handle, sql.cstring, -1.cint, addr st, nil) != SQLITE_OK:
    raiseDb(s, "prepare failed for " & sql)
  defer: discard sqlite3_finalize(st)
  bindParams(s, st, params)
  let rc = sqlite3_step(st)
  if rc != SQLITE_DONE and rc != SQLITE_ROW:
    raiseDb(s, "exec failed for " & sql)
  result = sqlite3_last_insert_rowid(s.handle)

proc exec(s: Store, sql: string, params: seq[JsonNode] = @[]): int64 =
  acquire(s.lock)
  defer: release(s.lock)
  result = execUnlocked(s, sql, params)

type
  SqlOperation = object
    sql: string
    params: seq[JsonNode]

proc execTransaction(s: Store, ops: openArray[SqlOperation]) =
  acquire(s.lock)
  defer: release(s.lock)
  execRawUnlocked(s, "BEGIN IMMEDIATE;")
  try:
    for op in ops:
      discard execUnlocked(s, op.sql, op.params)
    execRawUnlocked(s, "COMMIT;")
  except CatchableError:
    try: execRawUnlocked(s, "ROLLBACK;")
    except CatchableError: discard
    raise

proc getStr(r: Row, k: string, d = ""): string =
  if r.hasKey(k) and r[k].kind == JString: r[k].getStr() else: d

proc getInt(r: Row, k: string, d: int64 = 0): int64 =
  if r.hasKey(k) and r[k].kind == JString:
    try: parseBiggestInt(r[k].getStr()) except CatchableError: d
  else: d

proc getFloat(r: Row, k: string, d = 0.0): float =
  if r.hasKey(k) and r[k].kind == JString:
    try: parseFloat(r[k].getStr()) except CatchableError: d
  else: d

proc getJson(r: Row, k: string): JsonNode =
  let s = getStr(r, k, "")
  if s.len == 0: return newJObject()
  try: parseJson(s) except CatchableError: newJObject()

var
  fts5Available = false

proc migrate(s: Store) =
  s.execRaw("PRAGMA journal_mode=WAL;")
  s.execRaw("PRAGMA synchronous=NORMAL;")
  s.execRaw("PRAGMA foreign_keys=ON;")
  s.execRaw("PRAGMA temp_store=MEMORY;")
  s.execRaw("PRAGMA busy_timeout=15000;")
  s.execRaw("""
CREATE TABLE IF NOT EXISTS tenants (
  tenant_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  api_key_hash TEXT NOT NULL,
  token_budget INTEGER NOT NULL DEFAULT 100000000,
  tokens_used INTEGER NOT NULL DEFAULT 0,
  allowed_tools TEXT NOT NULL DEFAULT '[]',
  created_at REAL NOT NULL
);""")
  s.execRaw("""
CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  title TEXT NOT NULL,
  spec_json TEXT NOT NULL,
  initial_state_json TEXT NOT NULL,
  state_json TEXT NOT NULL,
  latest_obs_json TEXT NOT NULL,
  status TEXT NOT NULL,
  step_index INTEGER NOT NULL DEFAULT 0,
  max_steps INTEGER NOT NULL DEFAULT 1000,
  tokens_used INTEGER NOT NULL DEFAULT 0,
  terminal_reason TEXT NOT NULL DEFAULT '',
  verified INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);""")
  s.execRaw("CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant_id, status, updated_at DESC);")
  s.execRaw("""
CREATE TABLE IF NOT EXISTS checkpoints (
  ckpt_id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  tenant_id TEXT NOT NULL,
  step_index INTEGER NOT NULL,
  state_json TEXT NOT NULL,
  obs_json TEXT NOT NULL,
  action_json TEXT NOT NULL,
  patch_json TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  digest TEXT NOT NULL,
  created_at REAL NOT NULL,
  UNIQUE(task_id, step_index),
  FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);""")
  s.execRaw("""
CREATE TABLE IF NOT EXISTS raw_traces (
  trace_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  tenant_id TEXT NOT NULL,
  step_index INTEGER NOT NULL,
  initial_state_json TEXT NOT NULL,
  skill_id TEXT NOT NULL,
  action_json TEXT NOT NULL,
  obs_json TEXT NOT NULL,
  delta_json TEXT NOT NULL,
  post_state_json TEXT NOT NULL,
  success INTEGER NOT NULL,
  latency_ms INTEGER NOT NULL,
  receipt_json TEXT NOT NULL,
  immutable_hash TEXT NOT NULL UNIQUE,
  created_at REAL NOT NULL
);""")
  s.execRaw("""
CREATE TRIGGER IF NOT EXISTS raw_traces_no_update
BEFORE UPDATE ON raw_traces
BEGIN
  SELECT RAISE(ABORT, 'raw_traces is an immutable ledger');
END;""")
  s.execRaw("""
CREATE TRIGGER IF NOT EXISTS raw_traces_no_delete
BEFORE DELETE ON raw_traces
BEGIN
  SELECT RAISE(ABORT, 'raw_traces is an immutable ledger');
END;""")
  s.execRaw("""
CREATE TABLE IF NOT EXISTS skills (
  skill_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  name TEXT NOT NULL,
  domain TEXT NOT NULL,
  trigger_spec TEXT NOT NULL,
  procedure_spec TEXT NOT NULL,
  preconditions_json TEXT NOT NULL DEFAULT '[]',
  postconditions_json TEXT NOT NULL DEFAULT '[]',
  failure_modes_json TEXT NOT NULL DEFAULT '[]',
  version INTEGER NOT NULL DEFAULT 1,
  active INTEGER NOT NULL DEFAULT 1,
  success_count INTEGER NOT NULL DEFAULT 0,
  failure_count INTEGER NOT NULL DEFAULT 0,
  reward REAL NOT NULL DEFAULT 0.0,
  embedding_json TEXT NOT NULL DEFAULT '[]',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE(tenant_id, name)
);""")
  s.execRaw("CREATE INDEX IF NOT EXISTS idx_skills_active ON skills(tenant_id, active, reward DESC);")
  try:
    s.execRaw("CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(skill_id UNINDEXED, tenant_id UNINDEXED, name, domain, trigger_spec, procedure_spec, tokenize='porter ascii');")
    s.execRaw("""
CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
  INSERT INTO skill_fts(skill_id, tenant_id, name, domain, trigger_spec, procedure_spec)
  VALUES (new.skill_id, new.tenant_id, new.name, new.domain, new.trigger_spec, new.procedure_spec);
END;""")
    s.execRaw("""
CREATE TRIGGER IF NOT EXISTS skills_ad AFTER DELETE ON skills BEGIN
  DELETE FROM skill_fts WHERE skill_id = old.skill_id;
END;""")
    s.execRaw("""
CREATE TRIGGER IF NOT EXISTS skills_au AFTER UPDATE ON skills BEGIN
  DELETE FROM skill_fts WHERE skill_id = old.skill_id;
  INSERT INTO skill_fts(skill_id, tenant_id, name, domain, trigger_spec, procedure_spec)
  VALUES (new.skill_id, new.tenant_id, new.name, new.domain, new.trigger_spec, new.procedure_spec);
END;""")
    s.execRaw("DELETE FROM skill_fts;")
    s.execRaw("INSERT INTO skill_fts(skill_id, tenant_id, name, domain, trigger_spec, procedure_spec) SELECT skill_id, tenant_id, name, domain, trigger_spec, procedure_spec FROM skills;")
    fts5Available = true
  except CatchableError:
    fts5Available = false
  s.execRaw("""
CREATE TABLE IF NOT EXISTS cognition (
  cog_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  tenant_id TEXT NOT NULL,
  step_index INTEGER NOT NULL,
  vector_json TEXT NOT NULL,
  gate REAL NOT NULL,
  subgoal TEXT NOT NULL,
  strategy TEXT NOT NULL,
  created_at REAL NOT NULL
);""")
  s.execRaw("CREATE INDEX IF NOT EXISTS idx_cognition_task ON cognition(task_id, created_at DESC);")
  s.execRaw("""
CREATE TABLE IF NOT EXISTS reflections (
  reflection_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  tenant_id TEXT NOT NULL,
  patch_json TEXT NOT NULL,
  failure_point TEXT NOT NULL,
  pivot_action TEXT NOT NULL,
  attribution TEXT NOT NULL,
  verifier_report_json TEXT NOT NULL,
  created_at REAL NOT NULL
);""")
  s.execRaw("""
CREATE TABLE IF NOT EXISTS policy_weights (
  tenant_id TEXT NOT NULL,
  token TEXT NOT NULL,
  weight REAL NOT NULL,
  updates INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL,
  PRIMARY KEY(tenant_id, token)
);""")
  s.execRaw("""
CREATE TABLE IF NOT EXISTS knowledge_docs (
  doc_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  slug TEXT NOT NULL,
  category TEXT NOT NULL,
  path TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  git_commit_hash TEXT NOT NULL,
  created_at REAL NOT NULL,
  UNIQUE(tenant_id, slug)
);""")
  s.execRaw("""
CREATE TABLE IF NOT EXISTS diagnostics (
  diagnostic_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  domain TEXT NOT NULL,
  spec_json TEXT NOT NULL,
  expectation_json TEXT NOT NULL,
  created_at REAL NOT NULL
);""")

var
  store: Store
  rngLock: Lock
  globalRng: Rand

proc nowF(): float = epochTime()

proc newId(prefix: string): string =
  acquire(rngLock)
  defer: release(rngLock)
  var buf = newStringOfCap(24)
  const alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
  for _ in 0 ..< 20:
    buf.add(alphabet[globalRng.rand(alphabet.len - 1)])
  result = prefix & "_" & buf

proc sha1Hex(s: string): string =
  result = ""
  for b in sha1.secureHash(s).Sha1Digest:
    result.add(toHex(b.int, 2).toLowerAscii())

proc canonical(node: JsonNode): string =
  if node == nil: return "null"
  case node.kind
  of JObject:
    var keys: seq[string] = @[]
    for k, _ in node.fields: keys.add(k)
    keys.sort()
    var parts: seq[string] = @[]
    for k in keys:
      parts.add(escapeJson(k) & ":" & canonical(node.fields[k]))
    result = "{" & parts.join(",") & "}"
  of JArray:
    var parts: seq[string] = @[]
    for it in node.elems: parts.add(canonical(it))
    result = "[" & parts.join(",") & "]"
  of JString: result = escapeJson(node.getStr())
  of JInt: result = $node.getBiggestInt()
  of JFloat: result = formatFloat(node.getFloat(), ffDefault, 10)
  of JBool: result = (if node.getBool(): "true" else: "false")
  of JNull: result = "null"

proc digestOf(node: JsonNode): string = sha1Hex(canonical(node))

proc tokenizeText(s: string): seq[string] =
  result = @[]
  var cur = newStringOfCap(32)
  for ch in s:
    if ch.isAlphaNumeric() or ch == '_':
      cur.add(ch.toLowerAscii())
    else:
      if cur.len >= 2: result.add(cur)
      cur.setLen(0)
  if cur.len >= 2: result.add(cur)

const StopWords = ["the", "and", "for", "with", "that", "this", "from", "into",
                   "have", "has", "are", "was", "were", "not", "but", "you",
                   "your", "then", "than", "will", "can", "any", "all", "its"]

proc contentTerms(s: string): seq[string] =
  result = @[]
  for t in tokenizeText(s):
    if t notin StopWords: result.add(t)

proc textEmbedding(text: string, dims: int = EmbeddingDim): seq[float] =
  result = newSeq[float](dims)
  let terms = contentTerms(text)
  if terms.len == 0: return
  var counts = initCountTable[string]()
  for t in terms: counts.inc(t)
  for term, c in counts:
    var h1 = 2166136261'u32
    for ch in term:
      h1 = (h1 xor uint32(ch.ord)) * 16777619'u32
    var h2 = 0'u32
    for ch in term:
      h2 = h2 * 16777619'u32 + uint32(ch.ord)
    let idx = int(h1 mod uint32(dims))
    let sign = if (h2 and 1'u32) == 1'u32: 1.0 else: -1.0
    let w = ln(1.0 + c.float)
    result[idx] += sign * w
    let idx2 = int(h2 mod uint32(dims))
    result[idx2] += sign * w * 0.5
  var norm = 0.0
  for v in result: norm += v * v
  norm = sqrt(norm)
  if norm > 1e-12:
    for i in 0 ..< result.len: result[i] = result[i] / norm

proc cosineSimilarity(a, b: seq[float]): float =
  if a.len == 0 or b.len == 0 or a.len != b.len: return 0.0
  var dot = 0.0
  for i in 0 ..< a.len: dot += a[i] * b[i]
  result = dot

proc embToJson(v: seq[float]): string =
  var arr = newJArray()
  for x in v: arr.add(%x)
  result = $arr

proc jsonToEmb(s: string): seq[float] =
  result = @[]
  if s.len == 0: return
  try:
    let j = parseJson(s)
    if j.kind == JArray:
      for it in j.elems: result.add(it.getFloat())
  except CatchableError:
    result = @[]

proc rrfFuse(dense, sparse: seq[(string, float)], limit: int): seq[(string, float)] =
  var fused = initTable[string, float]()
  for i, item in dense:
    fused[item[0]] = fused.getOrDefault(item[0], 0.0) + 1.0 / (RrfK + (i + 1).float)
  for i, item in sparse:
    fused[item[0]] = fused.getOrDefault(item[0], 0.0) + 1.0 / (RrfK + (i + 1).float)
  var lst: seq[(string, float)] = @[]
  for k, v in fused: lst.add((k, v))
  lst.sort(proc(x, y: (string, float)): int = cmp(y[1], x[1]))
  if lst.len > limit: lst.setLen(limit)
  result = lst

proc stalenessEncoding(elapsed: float): JsonNode =
  result = newJArray()
  for k in 0 ..< 4:
    let freq = pow(10.0, float(k) * 2.0 / 8.0)
    result.add(%sin(elapsed / freq))
    result.add(%cos(elapsed / freq))

proc searchSkills(tenantId, queryText: string, limit: int): seq[Row] =
  let rows = store.query("SELECT * FROM skills WHERE tenant_id=? AND active=1", @[%tenantId])
  if rows.len == 0: return @[]
  let qEmb = textEmbedding(queryText)
  var dense: seq[(string, float)] = @[]
  var byId = initTable[string, Row]()
  for r in rows:
    let id = getStr(r, "skill_id")
    byId[id] = r
    let e = jsonToEmb(getStr(r, "embedding_json"))
    dense.add((id, cosineSimilarity(qEmb, e)))
  dense.sort(proc(a, b: (string, float)): int = cmp(b[1], a[1]))
  if dense.len > limit * 4: dense.setLen(limit * 4)

  var sparse: seq[(string, float)] = @[]
  if fts5Available:
    let terms = contentTerms(queryText)
    if terms.len > 0:
      let ftsQuery = terms.mapIt(it.replace("\"", "")).join(" OR ")
      let ftsRows = store.query(
        "SELECT skill_id, rank FROM skill_fts WHERE skill_fts MATCH ? AND tenant_id=? ORDER BY rank LIMIT ?",
        @[%ftsQuery, %tenantId, %(limit * 4)])
      for fr in ftsRows:
        sparse.add((getStr(fr, "skill_id"), -getFloat(fr, "rank", 0.0)))
  if sparse.len == 0:
    let qTerms = contentTerms(queryText).toHashSet()
    for r in rows:
      let id = getStr(r, "skill_id")
      let docTerms = contentTerms(getStr(r, "name") & " " & getStr(r, "domain") & " " &
                                  getStr(r, "trigger_spec") & " " & getStr(r, "procedure_spec")).toHashSet()
      let isect = intersection(qTerms, docTerms).len.float
      if isect > 0.0: sparse.add((id, isect))
    sparse.sort(proc(a, b: (string, float)): int = cmp(b[1], a[1]))
    if sparse.len > limit * 4: sparse.setLen(limit * 4)

  var policy = initTable[string, float]()
  let policyRows = store.query("SELECT token, weight FROM policy_weights WHERE tenant_id=? AND ABS(weight) > 0.000001", @[%tenantId])
  for pr in policyRows:
    policy[getStr(pr, "token").toLowerAscii()] = getFloat(pr, "weight", 0.0)

  let fused = rrfFuse(dense, sparse, limit * 2)
  var scored: seq[(Row, float)] = @[]
  for item in fused:
    if not byId.hasKey(item[0]): continue
    let r = byId[item[0]]
    let succ = getInt(r, "success_count", 0).float
    let fail = getInt(r, "failure_count", 0).float
    let prior = (succ + 1.0) / (succ + fail + 2.0)
    let reward = getFloat(r, "reward", 0.0)
    let corpus = getStr(r, "name") & " " & getStr(r, "domain") & " " & getStr(r, "trigger_spec") & " " & getStr(r, "procedure_spec")
    var pw = 0.0
    var seen = initHashSet[string]()
    for term in contentTerms(corpus):
      if term notin seen:
        seen.incl(term)
        pw += policy.getOrDefault(term, 0.0)
    scored.add((r, item[1] * 100.0 + prior * 2.0 + reward * 0.5 + clamp(pw, -8.0, 8.0) * 0.3))
  scored.sort(proc(a, b: (Row, float)): int = cmp(b[1], a[1]))
  result = @[]
  for i in 0 ..< min(limit, scored.len): result.add(scored[i][0])

proc searchKnowledge(tenantId, queryText: string, limit: int): seq[Row] =
  let rows = store.query("SELECT * FROM knowledge_docs WHERE tenant_id=?", @[%tenantId])
  if rows.len == 0: return @[]
  let qEmb = textEmbedding(queryText)
  let qTerms = contentTerms(queryText).toHashSet()
  let root = absolutePath(KnowledgeRoot / tenantId)
  let rootClean = if root.endsWith($DirSep): root else: root & $DirSep
  var byId = initTable[string, Row]()
  var dense: seq[(string, float)] = @[]
  var sparse: seq[(string, float)] = @[]
  for r0 in rows:
    var r = r0
    let id = getStr(r, "doc_id")
    var excerpt = ""
    let rel = getStr(r, "path")
    if rel.len > 0:
      let full = absolutePath(root / rel)
      if (full == root or full.startsWith(rootClean)) and fileExists(full):
        try:
          let raw = readFile(full)
          excerpt = if raw.len > 4096: raw[0 ..< 4096] else: raw
        except CatchableError:
          discard
    r["excerpt"] = %excerpt
    byId[id] = r
    let corpus = getStr(r, "slug") & " " & getStr(r, "category") & " " & excerpt
    dense.add((id, cosineSimilarity(qEmb, textEmbedding(corpus))))
    let dTerms = contentTerms(corpus).toHashSet()
    let overlap = intersection(qTerms, dTerms).len.float
    if overlap > 0.0: sparse.add((id, overlap))
  dense.sort(proc(a, b: (string, float)): int = cmp(b[1], a[1]))
  sparse.sort(proc(a, b: (string, float)): int = cmp(b[1], a[1]))
  if dense.len > limit * 4: dense.setLen(limit * 4)
  if sparse.len > limit * 4: sparse.setLen(limit * 4)
  let fused = rrfFuse(dense, sparse, max(limit, 1))
  result = @[]
  for item in fused:
    if byId.hasKey(item[0]): result.add(byId[item[0]])

type
  ToolResult = object
    ok: bool
    payload: JsonNode
    receipt: string
    message: string

proc safeJoin(tenant, rel: string): string =
  let workspaceBase = absolutePath(WorkspaceRoot)
  createDir(workspaceBase)
  if symlinkExists(workspaceBase): raise newException(ValueError, "workspace root cannot be a symbolic link")
  let base = absolutePath(workspaceBase / tenant)
  if symlinkExists(base): raise newException(ValueError, "tenant workspace cannot be a symbolic link")
  createDir(base)
  let baseClean = if base.endsWith($DirSep): base else: base & $DirSep
  var cleaned = rel.replace('\\', '/')
  while cleaned.startsWith("/"):
    if cleaned.len == 1: cleaned = ""
    else: cleaned = cleaned[1 .. ^1]
  var parts: seq[string] = @[]
  for seg in cleaned.split('/'):
    if seg.len == 0 or seg == ".": continue
    if seg == "..":
      if parts.len == 0: raise newException(ValueError, "path escapes workspace sandbox")
      parts.setLen(parts.len - 1)
      continue
    if '\0' in seg: raise newException(ValueError, "invalid path segment")
    parts.add(seg)
  var current = base
  for seg in parts:
    current = current / seg
    if symlinkExists(current): raise newException(ValueError, "symbolic links are not allowed in workspace paths")
  let cleanedPath = parts.join($DirSep)
  let full = if cleanedPath.len == 0: base else: absolutePath(baseClean / cleanedPath)
  if not (full == base or full.startsWith(baseClean)):
    raise newException(ValueError, "path escapes workspace sandbox")
  result = full

proc atomicWrite(full, content: string) =
  createDir(parentDir(full))
  let tmp = full & ".tmp." & $getTime().toUnix() & "." & newId("t")
  try:
    writeFile(tmp, content)
    moveFile(tmp, full)
  finally:
    if fileExists(tmp):
      try: removeFile(tmp)
      except CatchableError: discard

proc readLinesOf(full: string): seq[string] =
  if not fileExists(full): return @[]
  let raw = readFile(full)
  if raw.len == 0: return @[]
  result = raw.splitLines()
  if result.len > 0 and result[^1].len == 0 and raw.endsWith("\n"):
    result.setLen(result.len - 1)

proc evalMathExpression(expr: string): (bool, float, string) =
  var pos = 0
  var failed = false
  var errMsg = ""

  proc fail(msg: string): float =
    if not failed:
      failed = true
      errMsg = msg
    0.0

  proc skipWs() =
    while pos < expr.len and expr[pos] in {' ', '\t', '\r', '\n'}: inc pos

  proc parseExpr(): float

  proc parseNumber(): float =
    skipWs()
    let start = pos
    var sawDigit = false
    while pos < expr.len and expr[pos].isDigit():
      sawDigit = true
      inc pos
    if pos < expr.len and expr[pos] == '.':
      inc pos
      while pos < expr.len and expr[pos].isDigit():
        sawDigit = true
        inc pos
    if not sawDigit:
      return fail("invalid numeric literal at pos " & $start)
    if pos < expr.len and expr[pos] in {'e', 'E'}:
      let expStart = pos
      inc pos
      if pos < expr.len and expr[pos] in {'+', '-'}: inc pos
      let digitsStart = pos
      while pos < expr.len and expr[pos].isDigit(): inc pos
      if pos == digitsStart:
        pos = expStart
        return fail("invalid numeric exponent at pos " & $expStart)
    try:
      result = parseFloat(expr[start ..< pos])
    except CatchableError:
      result = fail("invalid numeric literal at pos " & $start)

  proc parsePrimary(): float =
    skipWs()
    if pos >= expr.len: return fail("unexpected end of expression")
    if expr[pos] == '(':
      inc pos
      let v = parseExpr()
      skipWs()
      if pos >= expr.len or expr[pos] != ')': return fail("missing closing parenthesis")
      inc pos
      return v
    if expr[pos].isAlphaAscii():
      var name = ""
      while pos < expr.len and (expr[pos].isAlphaNumeric() or expr[pos] == '_'):
        name.add(expr[pos].toLowerAscii())
        inc pos
      skipWs()
      if name == "pi" and (pos >= expr.len or expr[pos] != '('): return PI
      if name == "e" and (pos >= expr.len or expr[pos] != '('): return E
      if pos >= expr.len or expr[pos] != '(':
        return fail("unknown constant or symbol: " & name)
      inc pos
      let a = parseExpr()
      skipWs()
      if pos >= expr.len or expr[pos] != ')': return fail("missing closing parenthesis")
      inc pos
      if failed: return 0.0
      case name
      of "sin": return sin(a)
      of "cos": return cos(a)
      of "tan": return tan(a)
      of "sqrt":
        if a < 0.0: return fail("domain error: sqrt of negative")
        return sqrt(a)
      of "abs": return abs(a)
      of "ln":
        if a <= 0.0: return fail("domain error: ln non-positive")
        return ln(a)
      of "log10":
        if a <= 0.0: return fail("domain error: log10 non-positive")
        return log10(a)
      of "exp": return exp(a)
      of "floor": return floor(a)
      of "ceil": return ceil(a)
      of "round": return round(a)
      else: return fail("unknown function: " & name)
    if expr[pos].isDigit() or expr[pos] == '.':
      return parseNumber()
    fail("invalid token at pos " & $pos)

  proc parseUnary(): float

  proc parsePower(): float =
    var v = parsePrimary()
    if failed: return 0.0
    skipWs()
    if pos + 1 < expr.len and expr[pos] == '*' and expr[pos + 1] == '*':
      pos += 2
      let rhs = parseUnary()
      if failed: return 0.0
      v = pow(v, rhs)
    elif pos < expr.len and expr[pos] == '^':
      inc pos
      let rhs = parseUnary()
      if failed: return 0.0
      v = pow(v, rhs)
    v

  proc parseUnary(): float =
    skipWs()
    if pos < expr.len and expr[pos] == '+':
      inc pos
      return parseUnary()
    if pos < expr.len and expr[pos] == '-':
      inc pos
      return -parseUnary()
    parsePower()

  proc parseTerm(): float =
    var v = parseUnary()
    while not failed:
      skipWs()
      if pos < expr.len and expr[pos] == '*' and not (pos + 1 < expr.len and expr[pos + 1] == '*'):
        inc pos
        v *= parseUnary()
      elif pos < expr.len and expr[pos] == '/':
        inc pos
        let d = parseUnary()
        if abs(d) < 1e-15: return fail("division by zero")
        v /= d
      elif pos < expr.len and expr[pos] == '%':
        inc pos
        let d = parseUnary()
        if abs(d) < 1e-15: return fail("modulo by zero")
        v = v - d * floor(v / d)
      else:
        break
    v

  proc parseExpr(): float =
    var v = parseTerm()
    while not failed:
      skipWs()
      if pos < expr.len and expr[pos] == '+':
        inc pos
        v += parseTerm()
      elif pos < expr.len and expr[pos] == '-':
        inc pos
        v -= parseTerm()
      else:
        break
    v

  let value = parseExpr()
  skipWs()
  if not failed and pos != expr.len:
    discard fail("trailing unparsed token at pos " & $pos)
  if failed: result = (false, 0.0, errMsg)
  else: result = (true, value, "")

proc sanitizeKnowledgeName(s: string): string =
  result = newStringOfCap(s.len)
  for ch in s:
    if ch.isAlphaNumeric() or ch in {'-', '_', '.'}: result.add(ch)
    elif ch in {' ', '/', '\\', ':'}: result.add('-')
  while result.contains("--"): result = result.replace("--", "-")
  result = result.strip(chars = {'-', '.'})
  if result.len == 0: result = "entry"
  if result.len > 96: result.setLen(96)

proc ensureKnowledgeRepo(tenant: string): string =
  let safeTenant = sanitizeKnowledgeName(tenant)
  let dir = absolutePath(KnowledgeRoot / safeTenant)
  createDir(dir)
  if not dirExists(dir / ".git"):
    let initRes = execCmdEx("git -C " & quoteShell(dir) & " init")
    if initRes.exitCode != 0: raise newException(IOError, "git init failed: " & initRes.output)
    let emailRes = execCmdEx("git -C " & quoteShell(dir) & " config user.email agent@modular.runtime")
    if emailRes.exitCode != 0: raise newException(IOError, "git config failed: " & emailRes.output)
    let nameRes = execCmdEx("git -C " & quoteShell(dir) & " config user.name AutonomousRuntime")
    if nameRes.exitCode != 0: raise newException(IOError, "git config failed: " & nameRes.output)
  result = dir

proc commitKnowledgeDoc(tenant, slug, category, body: string): string =
  let repo = ensureKnowledgeRepo(tenant)
  let safeSlug = sanitizeKnowledgeName(slug)
  let safeCategory = sanitizeKnowledgeName(category)
  let rel = safeCategory & "_" & safeSlug & ".md"
  let full = repo / rel
  let chash = sha1Hex(body)
  let existing = store.query("SELECT doc_id, content_hash, git_commit_hash FROM knowledge_docs WHERE tenant_id=? AND slug=?", @[%tenant, %slug])
  if existing.len > 0 and getStr(existing[0], "content_hash") == chash and fileExists(full):
    return getStr(existing[0], "doc_id")
  atomicWrite(full, "# " & slug & "\nCategory: " & category & "\n\n" & body & "\n")
  let addRes = execCmdEx("git -C " & quoteShell(repo) & " add -- " & quoteShell(rel))
  if addRes.exitCode != 0: raise newException(IOError, "git add failed: " & addRes.output)
  let commitRes = execCmdEx("git -C " & quoteShell(repo) & " commit -m " & quoteShell("knowledge update: " & safeSlug))
  if commitRes.exitCode != 0:
    let statusRes = execCmdEx("git -C " & quoteShell(repo) & " status --porcelain -- " & quoteShell(rel))
    if statusRes.exitCode != 0 or statusRes.output.strip().len > 0:
      raise newException(IOError, "git commit failed: " & commitRes.output)
  let revRes = execCmdEx("git -C " & quoteShell(repo) & " rev-parse HEAD")
  if revRes.exitCode != 0: raise newException(IOError, "git rev-parse failed: " & revRes.output)
  let commitHash = revRes.output.strip()
  let did = if existing.len > 0: getStr(existing[0], "doc_id") else: newId("doc")
  discard store.exec(
    "INSERT INTO knowledge_docs (doc_id, tenant_id, slug, category, path, content_hash, git_commit_hash, created_at) " &
    "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(tenant_id, slug) DO UPDATE SET " &
    "category=excluded.category, path=excluded.path, content_hash=excluded.content_hash, " &
    "git_commit_hash=excluded.git_commit_hash, created_at=excluded.created_at",
    @[%did, %tenant, %slug, %category, %rel, %chash, %commitHash, %nowF()])
  result = did

type
  ToolHandler = proc(tenant: string, args: JsonNode): Future[ToolResult] {.closure, gcsafe.}
  ToolSpec = object
    name: string
    description: string
    schema: JsonNode
    handler: ToolHandler

var toolRegistry: OrderedTable[string, ToolSpec] = initOrderedTable[string, ToolSpec]()

proc registerTool(name, description: string, schema: JsonNode, handler: ToolHandler) =
  toolRegistry[name] = ToolSpec(name: name, description: description, schema: schema, handler: handler)

proc parseLegacyNumber(part: string, ok: var bool): uint64 =
  if part.len == 0:
    ok = false
    return 0
  var base = 10'u64
  var i = 0
  if part.len > 2 and part[0] == '0' and part[1] in {'x', 'X'}:
    base = 16
    i = 2
  elif part.len > 1 and part[0] == '0':
    base = 8
    i = 1
  if i >= part.len:
    ok = true
    return 0
  var value = 0'u64
  while i < part.len:
    let ch = part[i]
    var d = -1
    if ch in {'0'..'9'}: d = ch.ord - '0'.ord
    elif ch in {'a'..'f'}: d = 10 + ch.ord - 'a'.ord
    elif ch in {'A'..'F'}: d = 10 + ch.ord - 'A'.ord
    if d < 0 or uint64(d) >= base or value > (high(uint32).uint64 - uint64(d)) div base:
      ok = false
      return 0
    value = value * base + uint64(d)
    inc i
  ok = true
  value

proc parseLegacyIpv4(host: string): (bool, array[4, uint8]) =
  let parts = host.split('.')
  if parts.len < 1 or parts.len > 4: return (false, default(array[4, uint8]))
  var nums: seq[uint64] = @[]
  for part in parts:
    var ok = false
    let n = parseLegacyNumber(part, ok)
    if not ok: return (false, default(array[4, uint8]))
    nums.add(n)
  var value = 0'u64
  case nums.len
  of 1:
    if nums[0] > 0xffffffff'u64: return (false, default(array[4, uint8]))
    value = nums[0]
  of 2:
    if nums[0] > 0xff'u64 or nums[1] > 0xffffff'u64: return (false, default(array[4, uint8]))
    value = (nums[0] shl 24) or nums[1]
  of 3:
    if nums[0] > 0xff'u64 or nums[1] > 0xff'u64 or nums[2] > 0xffff'u64: return (false, default(array[4, uint8]))
    value = (nums[0] shl 24) or (nums[1] shl 16) or nums[2]
  of 4:
    for n in nums:
      if n > 0xff'u64: return (false, default(array[4, uint8]))
    value = (nums[0] shl 24) or (nums[1] shl 16) or (nums[2] shl 8) or nums[3]
  else:
    return (false, default(array[4, uint8]))
  var outv: array[4, uint8]
  outv[0] = uint8((value shr 24) and 0xff)
  outv[1] = uint8((value shr 16) and 0xff)
  outv[2] = uint8((value shr 8) and 0xff)
  outv[3] = uint8(value and 0xff)
  (true, outv)

proc blockedIpv4(a: array[4, uint8]): bool =
  let x = a[0].int
  let y = a[1].int
  if x == 0 or x == 10 or x == 127: return true
  if x == 100 and y >= 64 and y <= 127: return true
  if x == 169 and y == 254: return true
  if x == 172 and y >= 16 and y <= 31: return true
  if x == 192 and y == 168: return true
  if x == 198 and y in [18, 19]: return true
  if x >= 224: return true
  false

proc blockedIp(ip: IpAddress): bool =
  case ip.family
  of IpAddressFamily.IPv4:
    blockedIpv4(ip.address_v4)
  of IpAddressFamily.IPv6:
    let a = ip.address_v6
    var allZero = true
    for b in a:
      if b != 0'u8: allZero = false
    if allZero: return true
    var loopback = true
    for i in 0 ..< 15:
      if a[i] != 0'u8: loopback = false
    if loopback and a[15] == 1'u8: return true
    if (a[0] and 0xfe'u8) == 0xfc'u8: return true
    if a[0] == 0xfe'u8 and (a[1] and 0xc0'u8) == 0x80'u8: return true
    if a[0] == 0xff'u8: return true
    var mapped = true
    for i in 0 ..< 10:
      if a[i] != 0'u8: mapped = false
    if mapped and a[10] == 0xff'u8 and a[11] == 0xff'u8:
      let v4 = [a[12], a[13], a[14], a[15]]
      return blockedIpv4(v4)
    var compatible = true
    for i in 0 ..< 12:
      if a[i] != 0'u8: compatible = false
    if compatible:
      let v4 = [a[12], a[13], a[14], a[15]]
      if blockedIpv4(v4): return true
    if a[0] == 0x20'u8 and a[1] == 0x02'u8:
      let embedded = [a[2], a[3], a[4], a[5]]
      if blockedIpv4(embedded): return true
    if a[0] == 0x20'u8 and a[1] == 0x01'u8 and a[2] == 0x00'u8 and a[3] == 0x00'u8: return true
    false

proc validateOutboundHost(hostInput: string) =
  var host = hostInput.toLowerAscii().strip()
  while host.len > 0 and host.endsWith("."):
    host.setLen(host.len - 1)
  if host.len == 0: raise newException(ValueError, "URL hostname required")
  if '%' in host or '\0' in host: raise newException(ValueError, "invalid URL hostname")
  if host == "localhost" or host.endsWith(".localhost") or host == "metadata.google.internal" or host.endsWith(".metadata.google.internal"):
    raise newException(ValueError, "SSRF guard: target blocked")
  let (legacyOk, legacyIp) = parseLegacyIpv4(host)
  if legacyOk:
    if blockedIpv4(legacyIp): raise newException(ValueError, "SSRF guard: target blocked")
    return
  if isIpAddress(host):
    let ip = parseIpAddress(host)
    if blockedIp(ip): raise newException(ValueError, "SSRF guard: target blocked")
    return
  var info: ptr AddrInfo = nil
  try:
    info = getAddrInfo(host, Port(80), AF_UNSPEC, SOCK_STREAM, IPPROTO_TCP)
    if info == nil: raise newException(ValueError, "hostname did not resolve")
    var cur = info
    var resolved = 0
    while cur != nil:
      if cur.ai_addr != nil:
        let addrText = getAddrString(cur.ai_addr)
        if isIpAddress(addrText):
          inc resolved
          if blockedIp(parseIpAddress(addrText)):
            raise newException(ValueError, "SSRF guard: resolved target blocked")
      cur = cur.ai_next
    if resolved == 0: raise newException(ValueError, "hostname did not resolve to an IP address")
  finally:
    if info != nil: freeAddrInfo(info)

proc validateOutboundUrl(url: string): Uri =
  if '\r' in url or '\n' in url: raise newException(ValueError, "invalid URL")
  let parsed = parseUri(url)
  let scheme = parsed.scheme.toLowerAscii()
  if scheme notin ["http", "https"]: raise newException(ValueError, "only http/https allowed")
  if parsed.username.len > 0 or parsed.password.len > 0: raise newException(ValueError, "userinfo in URL is not allowed")
  validateOutboundHost(parsed.hostname)
  parsed

proc awaitBounded[T](fut: Future[T], timeoutMs: int, label: string): Future[T] {.async.} =
  if not await withTimeout(fut, timeoutMs):
    raise newException(TimeoutError, label & " timed out")
  return fut.read()

proc readBoundedBody(resp: AsyncResponse, maxBytes: int): Future[(string, bool)] {.async.} =
  var body = newStringOfCap(min(maxBytes, 8192))
  var truncated = false
  while true:
    let readFuture = resp.bodyStream.read()
    let item = await awaitBounded(readFuture, HttpTimeoutMs, "HTTP response body")
    if not item[0]: break
    let chunk = item[1]
    if body.len + chunk.len <= maxBytes:
      body.add(chunk)
    else:
      let remain = maxBytes - body.len
      if remain > 0: body.add(chunk[0 ..< remain])
      truncated = true
      break
  return (body, truncated)

proc responseHeadersJson(headers: HttpHeaders): JsonNode =
  result = newJObject()
  for k, v in headers:
    result[k] = %v

proc mapHttpMethod(methodStr: string): HttpMethod =
  case methodStr.toUpperAscii()
  of "POST": HttpPost
  of "PUT": HttpPut
  of "DELETE": HttpDelete
  of "HEAD": HttpHead
  of "PATCH": HttpPatch
  of "OPTIONS": HttpOptions
  else: HttpGet

proc initTools() =
  registerTool("write_file", "Atomically write content to a sandboxed workspace file.",
    %*{"path": "string", "content": "string"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let rel = args{"path"}.getStr("")
      let content = args{"content"}.getStr("")
      if rel.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "path required")
      let full = safeJoin(tenant, rel)
      atomicWrite(full, content)
      return ToolResult(ok: true,
        payload: %*{"path": rel, "bytes_written": content.len, "sha1": sha1Hex(content)},
        receipt: "write:" & rel & ":" & sha1Hex(content),
        message: "wrote " & $content.len & " bytes"))

  registerTool("read_file", "Read file lines or bounded bytes from workspace.",
    %*{"path": "string", "start_line": "int optional", "end_line": "int optional", "max_bytes": "int optional"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let rel = args{"path"}.getStr("")
      if rel.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "path required")
      let full = safeJoin(tenant, rel)
      if not fileExists(full): return ToolResult(ok: false, payload: %*{"path": rel}, message: "file not found")
      let lines = readLinesOf(full)
      let sLine = args{"start_line"}.getInt(1)
      let eLine = args{"end_line"}.getInt(0)
      let a = max(sLine - 1, 0)
      let b = if eLine > 0: min(eLine, lines.len) else: lines.len
      var body = if a < b: lines[a ..< b].join("\n") else: ""
      let maxBytes = max(1, min(args{"max_bytes"}.getInt(MaxHttpBodyBytes), MaxHttpBodyBytes))
      var truncated = false
      if body.len > maxBytes:
        body = body[0 ..< maxBytes]
        truncated = true
      return ToolResult(ok: true,
        payload: %*{"path": rel, "content": body, "total_lines": lines.len, "bytes_read": body.len, "truncated": truncated, "sha1": sha1Hex(body)},
        receipt: "read:" & rel & ":" & $lines.len,
        message: "read " & $body.len & " bytes"))

  registerTool("append_file", "Atomically append lines with optional deduplication.",
    %*{"path": "string", "lines": "string[]", "unique": "bool default true"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let rel = args{"path"}.getStr("")
      if rel.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "path required")
      var incoming: seq[string] = @[]
      if args.hasKey("lines") and args["lines"].kind == JArray:
        for it in args["lines"].elems: incoming.add(it.getStr())
      elif args{"content"}.getStr("").len > 0:
        incoming = args{"content"}.getStr().splitLines()
      if incoming.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "lines or content required")
      let unique = args{"unique"}.getBool(true)
      let full = safeJoin(tenant, rel)
      var existing = readLinesOf(full)
      var seen = initHashSet[string]()
      if unique:
        for l in existing: seen.incl(l)
      var added = 0
      var skipped = 0
      for l in incoming:
        if unique and l in seen:
          inc skipped
          continue
        existing.add(l)
        if unique: seen.incl(l)
        inc added
      let finalContent = existing.join("\n") & (if existing.len > 0: "\n" else: "")
      atomicWrite(full, finalContent)
      return ToolResult(ok: true,
        payload: %*{"path": rel, "added_count": added, "skipped_count": skipped, "total_lines": existing.len, "sha1": sha1Hex(finalContent)},
        receipt: "append:" & rel & ":" & $added,
        message: "appended " & $added & " lines, skipped " & $skipped))

  registerTool("replace_lines", "Replace 1-based inclusive line range in file.",
    %*{"path": "string", "start_line": "int", "end_line": "int", "lines": "string[]"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let rel = args{"path"}.getStr("")
      if rel.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "path required")
      let sLine = args{"start_line"}.getInt(0)
      let eLine = args{"end_line"}.getInt(0)
      if sLine < 1 or eLine < sLine - 1:
        return ToolResult(ok: false, payload: newJObject(), message: "invalid line boundaries")
      let full = safeJoin(tenant, rel)
      var lines = readLinesOf(full)
      if sLine > lines.len + 1:
        return ToolResult(ok: false, payload: newJObject(), message: "start_line beyond EOF")
      var replacements: seq[string] = @[]
      if args.hasKey("lines") and args["lines"].kind == JArray:
        for it in args["lines"].elems: replacements.add(it.getStr())
      elif args{"content"}.getStr("").len > 0:
        replacements = args{"content"}.getStr().splitLines()
      let a = sLine - 1
      let b = min(eLine, lines.len)
      var nextLines: seq[string] = @[]
      for i in 0 ..< a: nextLines.add(lines[i])
      for r in replacements: nextLines.add(r)
      for i in b ..< lines.len: nextLines.add(lines[i])
      let finalContent = nextLines.join("\n") & (if nextLines.len > 0: "\n" else: "")
      atomicWrite(full, finalContent)
      return ToolResult(ok: true,
        payload: %*{"path": rel, "removed_count": max(0, b - a), "inserted_count": replacements.len, "total_lines": nextLines.len, "sha1": sha1Hex(finalContent)},
        receipt: "replace:" & rel & ":" & $sLine & "-" & $eLine,
        message: "replaced lines " & $sLine & ".." & $eLine))

  registerTool("check_lines", "Batch exact line presence check in workspace file.",
    %*{"path": "string", "lines": "string[]"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let rel = args{"path"}.getStr("")
      if rel.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "path required")
      var probes: seq[string] = @[]
      if args.hasKey("lines") and args["lines"].kind == JArray:
        for it in args["lines"].elems: probes.add(it.getStr())
      if probes.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "lines array required")
      let full = safeJoin(tenant, rel)
      let lines = readLinesOf(full)
      var present = initTable[string, int]()
      for i, l in lines:
        if not present.hasKey(l): present[l] = i + 1
      var results = newJArray()
      var missing = 0
      for p in probes:
        let ok = present.hasKey(p)
        if not ok: inc missing
        results.add(%*{"line": p, "present": ok, "line_number": (if ok: present[p] else: 0)})
      return ToolResult(ok: true,
        payload: %*{"path": rel, "results": results, "missing_count": missing, "total_checked": probes.len},
        receipt: "check:" & rel & ":" & $probes.len,
        message: $(probes.len - missing) & "/" & $probes.len & " lines present"))

  registerTool("search_files", "Search workspace files recursively for substring pattern.",
    %*{"pattern": "string", "path": "string optional"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let pattern = args{"pattern"}.getStr("")
      if pattern.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "pattern required")
      let relDir = args{"path"}.getStr(".")
      let base = safeJoin(tenant, ".")
      let searchRoot = safeJoin(tenant, relDir)
      var hits = newJArray()
      var scanned = 0
      let needle = pattern.toLowerAscii()
      for p in walkDirRec(searchRoot):
        if not fileExists(p): continue
        inc scanned
        if scanned > 2000: break
        try:
          if getFileSize(p) > 2_000_000: continue
          let c = readFile(p)
          let rPath = relativePath(p, base).replace('\\', '/')
          var lineNo = 0
          for l in c.splitLines():
            inc lineNo
            if needle in l.toLowerAscii():
              hits.add(%*{"path": rPath, "line": lineNo, "text": l})
              if hits.len >= 100: break
        except CatchableError: discard
        if hits.len >= 100: break
      return ToolResult(ok: true,
        payload: %*{"pattern": pattern, "hits": hits, "scanned_files": scanned},
        receipt: "search:" & sha1Hex(pattern),
        message: "found " & $hits.len & " matches"))

  registerTool("list_dir", "List entries in a workspace directory.",
    %*{"path": "string optional"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let rel = args{"path"}.getStr(".")
      let target = safeJoin(tenant, rel)
      if not dirExists(target): return ToolResult(ok: false, payload: newJObject(), message: "directory not found")
      var entries = newJArray()
      for kind, p in walkDir(target):
        let name = extractFilename(p)
        var sz = 0'i64
        if kind == pcFile:
          try: sz = getFileSize(p) except CatchableError: sz = 0
        entries.add(%*{"name": name, "type": (if kind in {pcDir, pcLinkToDir}: "dir" else: "file"), "bytes": sz})
      return ToolResult(ok: true,
        payload: %*{"path": rel, "entries": entries},
        receipt: "list:" & rel,
        message: "listed " & $entries.len & " entries"))

  registerTool("delete_file", "Safely delete a workspace file.",
    %*{"path": "string"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let rel = args{"path"}.getStr("")
      if rel.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "path required")
      let full = safeJoin(tenant, rel)
      if not fileExists(full): return ToolResult(ok: false, payload: newJObject(), message: "file not found")
      removeFile(full)
      return ToolResult(ok: true, payload: %*{"path": rel, "deleted": true}, receipt: "del:" & rel, message: "deleted"))

  registerTool("memory_search", "Hybrid dense/sparse RRF search over skills and knowledge wiki.",
    %*{"query": "string", "limit": "int optional"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let q = args{"query"}.getStr("")
      if q.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "query required")
      let limit = max(1, min(args{"limit"}.getInt(5), 20))
      let skills = searchSkills(tenant, q, limit)
      var sArr = newJArray()
      for s in skills:
        sArr.add(%*{"skill_id": getStr(s, "skill_id"), "name": getStr(s, "name"),
                    "domain": getStr(s, "domain"), "trigger": getStr(s, "trigger_spec"),
                    "procedure": getStr(s, "procedure_spec"), "reward": getFloat(s, "reward", 0.0)})
      let docs = searchKnowledge(tenant, q, limit)
      var dArr = newJArray()
      for d in docs:
        dArr.add(%*{"slug": getStr(d, "slug"), "category": getStr(d, "category"), "path": getStr(d, "path"), "excerpt": getStr(d, "excerpt")})
      return ToolResult(ok: true,
        payload: %*{"query": q, "skills": sArr, "knowledge": dArr},
        receipt: "memsearch:" & sha1Hex(q),
        message: "retrieved " & $sArr.len & " skills, " & $dArr.len & " docs"))

  registerTool("memory_write", "Persist structured operational knowledge into the Git-backed playbook.",
    %*{"slug": "string", "category": "string optional", "body": "string"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let slug = args{"slug"}.getStr("")
      let body = args{"body"}.getStr("")
      if slug.len == 0 or body.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "slug and body required")
      let category = args{"category"}.getStr("operational")
      let did = commitKnowledgeDoc(tenant, slug, category, body)
      return ToolResult(ok: true,
        payload: %*{"doc_id": did, "slug": slug, "category": category},
        receipt: "memwrite:" & did,
        message: "persisted doc: " & slug))

  registerTool("math_eval", "Deterministic recursive-descent mathematical parser evaluation.",
    %*{"expression": "string"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      let expr = args{"expression"}.getStr("")
      if expr.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "expression required")
      let (ok, val, err) = evalMathExpression(expr)
      if not ok: return ToolResult(ok: false, payload: %*{"expression": expr}, message: err)
      return ToolResult(ok: true,
        payload: %*{"expression": expr, "value": val},
        receipt: "math:" & sha1Hex(expr),
        message: "= " & formatFloat(val, ffDefault, 10)))

  registerTool("http_fetch", "SSRF-guarded outbound HTTP request.",
    %*{"url": "string", "method": "string optional", "headers": "object optional", "body": "string optional"},
    proc(tenant: string, args: JsonNode): Future[ToolResult] {.async.} =
      var currentUrl = args{"url"}.getStr("")
      if currentUrl.len == 0: return ToolResult(ok: false, payload: newJObject(), message: "url required")
      var meth = mapHttpMethod(args{"method"}.getStr("GET"))
      var requestBody = args{"body"}.getStr("")
      var hdrs = newHttpHeaders()
      if args.hasKey("headers") and args["headers"].kind == JObject:
        for k, v in args["headers"].fields:
          let lk = k.toLowerAscii()
          if lk notin ["host", "content-length", "connection", "transfer-encoding"]:
            hdrs[k] = v.getStr()
      var redirects = 0
      try:
        while true:
          let parsed = validateOutboundUrl(currentUrl)
          var client = newAsyncHttpClient(maxRedirects = 0)
          try:
            let reqFuture = client.request(currentUrl, httpMethod = meth, body = requestBody, headers = hdrs)
            let resp = await awaitBounded(reqFuture, HttpTimeoutMs, "HTTP request")
            let status = resp.code.int
            if status in [301, 302, 303, 307, 308] and resp.headers.hasKey("location"):
              if redirects >= 5:
                return ToolResult(ok: false, payload: %*{"url": currentUrl, "status": status}, message: "too many redirects")
              let location = resp.headers["location"]
              let nextUrl = $combine(parsed, parseUri(location))
              let nextParsed = validateOutboundUrl(nextUrl)
              if nextParsed.hostname.toLowerAscii() != parsed.hostname.toLowerAscii() or nextParsed.port != parsed.port or nextParsed.scheme.toLowerAscii() != parsed.scheme.toLowerAscii():
                for secretHeader in ["Authorization", "Proxy-Authorization", "Cookie"]:
                  if hdrs.hasKey(secretHeader): hdrs.del(secretHeader)
              currentUrl = nextUrl
              inc redirects
              if status == 303 or ((status == 301 or status == 302) and meth == HttpPost):
                meth = HttpGet
                requestBody = ""
              continue
            let (resBody, truncated) = await readBoundedBody(resp, MaxHttpBodyBytes)
            let outHeaders = responseHeadersJson(resp.headers)
            return ToolResult(ok: status < 400,
              payload: %*{"url": currentUrl, "status": status, "headers": outHeaders, "body": resBody, "truncated": truncated},
              receipt: "http:" & sha1Hex(currentUrl & $status & resBody),
              message: "status " & $status)
          finally:
            client.close()
      except CatchableError as e:
        return ToolResult(ok: false, payload: %*{"url": currentUrl}, message: "fetch error: " & e.msg))

proc toolCatalog(allowed: HashSet[string]): JsonNode =
  result = newJArray()
  for name, spec in toolRegistry:
    if name notin allowed: continue
    result.add(%*{"name": name, "description": spec.description, "arguments": spec.schema})

proc countNodes(n: JsonNode): int =
  if n == nil: return 0
  result = 1
  case n.kind
  of JObject:
    for _, v in n.fields: result += countNodes(v)
  of JArray:
    for it in n.elems: result += countNodes(it)
  else: discard

proc trimArrayTail(node: JsonNode, key: string, limit: int) =
  if node.hasKey(key) and node[key].kind == JArray and node[key].elems.len > limit:
    let src = node[key]
    var trimmed = newJArray()
    for i in src.elems.len - limit ..< src.elems.len: trimmed.add(src[i])
    node[key] = trimmed

proc pruneSigma(sigma: JsonNode) =
  if sigma == nil or sigma.kind != JObject: return
  if sigma.hasKey("subgoals") and sigma["subgoals"].kind == JArray:
    var active = newJArray()
    var doneRecent: seq[JsonNode] = @[]
    for it in sigma["subgoals"].elems:
      let status = if it.kind == JObject: it{"status"}.getStr("open") else: "open"
      if status in ["open", "in_progress", "blocked"]:
        active.add(it)
    for i in countdown(sigma["subgoals"].elems.len - 1, 0):
      let it = sigma["subgoals"].elems[i]
      if it.kind == JObject and it{"status"}.getStr("open") == "done":
        doneRecent.add(it)
        if doneRecent.len >= 8: break
    var keep = newJArray()
    let doneSlots = max(0, min(8, 48 - active.elems.len))
    if doneSlots > 0:
      for i in countdown(doneSlots - 1, 0):
        if i < doneRecent.len: keep.add(doneRecent[i])
    for it in active.elems: keep.add(it)
    sigma["subgoals"] = keep
  trimArrayTail(sigma, "blockers", 16)
  trimArrayTail(sigma, "artifacts", 64)
  if sigma.hasKey("facts") and sigma["facts"].kind == JObject and sigma["facts"].fields.len > 128:
    var keys: seq[string] = @[]
    for k, _ in sigma["facts"].fields: keys.add(k)
    var newFacts = newJObject()
    let start = max(0, keys.len - 128)
    for i in start ..< keys.len: newFacts[keys[i]] = sigma["facts"][keys[i]]
    sigma["facts"] = newFacts
  while (countNodes(sigma) > 4000 or canonical(sigma).len > MaxStateBytes):
    var changed = false
    if sigma.hasKey("facts") and sigma["facts"].kind == JObject and sigma["facts"].fields.len > 0:
      for k, _ in sigma["facts"].fields:
        sigma["facts"].delete(k)
        changed = true
        break
    elif sigma.hasKey("artifacts") and sigma["artifacts"].kind == JArray and sigma["artifacts"].elems.len > 0:
      sigma["artifacts"].delete(0)
      changed = true
    elif sigma.hasKey("blockers") and sigma["blockers"].kind == JArray and sigma["blockers"].elems.len > 0:
      sigma["blockers"].delete(0)
      changed = true
    elif sigma.hasKey("subgoals") and sigma["subgoals"].kind == JArray:
      for i in 0 ..< sigma["subgoals"].elems.len:
        let it = sigma["subgoals"].elems[i]
        if it.kind == JObject and it{"status"}.getStr("") == "done":
          sigma["subgoals"].delete(i)
          changed = true
          break
    if not changed: break

proc deepMerge(base, patch: JsonNode): JsonNode =
  if patch == nil or patch.kind != JObject: return copy(base)
  result = if base != nil and base.kind == JObject: copy(base) else: newJObject()
  for k, v in patch.fields:
    if v.kind == JNull:
      if result.hasKey(k): result.delete(k)
    elif v.kind == JObject and result.hasKey(k) and result[k].kind == JObject:
      result[k] = deepMerge(result[k], v)
    else:
      result[k] = copy(v)

proc collectForbiddenKeys(n: JsonNode, path: string, errs: var seq[string]) =
  if n == nil: return
  case n.kind
  of JObject:
    for k, v in n.fields:
      let nextPath = if path.len == 0: "/" & k else: path & "/" & k
      if k in ["history", "transcript", "messages", "chain_of_thought", "scratchpad"]:
        errs.add("forbidden historical key at " & nextPath)
      collectForbiddenKeys(v, nextPath, errs)
  of JArray:
    for i, it in n.elems: collectForbiddenKeys(it, path & "/" & $i, errs)
  else: discard

proc validatePatch(patch: JsonNode): (bool, seq[string]) =
  var errs: seq[string] = @[]
  if patch == nil or patch.kind != JObject:
    return (false, @["state patch must be a JSON object"])
  collectForbiddenKeys(patch, "", errs)
  if countNodes(patch) > 4000: errs.add("state patch exceeds node complexity bound")
  if canonical(patch).len > MaxStateBytes: errs.add("state patch exceeds byte bound")
  (errs.len == 0, errs)

proc validateSigma(sigma: JsonNode): (bool, seq[string]) =
  var errs: seq[string] = @[]
  if sigma == nil or sigma.kind != JObject:
    return (false, @["state must be a valid JSON object"])
  for req in ["goal", "progress", "phase", "subgoals", "constraints", "facts", "blockers"]:
    if not sigma.hasKey(req): errs.add("missing required state field: " & req)
  if sigma.hasKey("progress"):
    let p = sigma["progress"]
    if p.kind notin {JInt, JFloat} or p.getFloat() < 0.0 or p.getFloat() > 1.0:
      errs.add("progress must be a numeric float in [0.0, 1.0]")
  collectForbiddenKeys(sigma, "", errs)
  if sigma.hasKey("subgoals") and sigma["subgoals"].kind == JArray and sigma["subgoals"].elems.len > 48:
    errs.add("state contains more than 48 active/recent subgoals")
  let sz = canonical(sigma).len
  if sz > MaxStateBytes: errs.add("state exceeds byte bound: " & $sz & " > " & $MaxStateBytes)
  let nodes = countNodes(sigma)
  if nodes > 4000: errs.add("state exceeds node complexity bound: " & $nodes)
  result = (errs.len == 0, errs)

proc defaultSigma(goal: string): JsonNode =
  result = %*{
    "goal": goal,
    "progress": 0.0,
    "phase": "bootstrap",
    "subgoals": newJArray(),
    "constraints": newJArray(),
    "facts": newJObject(),
    "blockers": newJArray(),
    "artifacts": newJArray(),
    "step_summary": "initialized",
    "system1": {"queued_actions": newJArray()}
  }

type
  LogprobItem = object
    token: string
    logprob: float
    textOffset: int

  LlmResponse = object
    content: string
    promptTokens: int
    completionTokens: int
    totalTokens: int
    logprobs: seq[LogprobItem]

  CompletionResponse = object
    text: string
    promptTokens: int
    completionTokens: int
    totalTokens: int
    logprobs: seq[LogprobItem]

proc apiKey(): string = getEnv("MODULAR_API_KEY", "")

proc enforcePromptBound(messages: JsonNode) =
  let n = canonical(messages).len
  if n > MaxPromptBytes:
    raise newException(ValueError, "prompt exceeds byte bound: " & $n & " > " & $MaxPromptBytes)

proc callChatCompletionsAsync(messages: JsonNode, maxTokens: int = MaxTokens,
                              temperature: float = Temperature,
                              jsonMode: bool = false,
                              logprobs: bool = false): Future[LlmResponse] {.async.} =
  let key = apiKey()
  if key.len == 0: raise newException(IOError, "MODULAR_API_KEY not configured")
  enforcePromptBound(messages)
  var body = %*{
    "model": ModularModel,
    "messages": messages,
    "stream": false,
    "temperature": temperature,
    "top_p": TopP,
    "max_tokens": max(1, min(maxTokens, MaxTokens)),
    "frequency_penalty": FrequencyPenalty,
    "presence_penalty": PresencePenalty,
    "seed": Seed
  }
  if jsonMode: body["response_format"] = %*{"type": "json_object"}
  if logprobs:
    body["logprobs"] = %true
    body["top_logprobs"] = %20
  var client = newAsyncHttpClient(maxRedirects = 0)
  defer: client.close()
  client.headers = newHttpHeaders({
    "Authorization": "Bearer " & key,
    "Content-Type": "application/json",
    "Accept": "application/json"
  })
  let requestFuture = client.request(ModularBaseUrl & "/chat/completions", httpMethod = HttpPost, body = $body)
  let resp = await awaitBounded(requestFuture, HttpTimeoutMs, "model request")
  let bodyFuture = resp.body()
  let raw = await awaitBounded(bodyFuture, HttpTimeoutMs, "model response body")
  if resp.code.int < 200 or resp.code.int >= 300:
    raise newException(IOError, "upstream model status " & $resp.code.int & ": " & (if raw.len > 4096: raw[0 ..< 4096] else: raw))
  let parsed = parseJson(raw)
  var outResp = LlmResponse()
  if parsed.hasKey("choices") and parsed["choices"].kind == JArray and parsed["choices"].elems.len > 0:
    let ch = parsed["choices"][0]
    if ch.hasKey("message") and ch["message"].kind == JObject:
      outResp.content = ch["message"]{"content"}.getStr("")
    if ch.hasKey("logprobs") and ch["logprobs"].kind == JObject and ch["logprobs"].hasKey("content"):
      let lContent = ch["logprobs"]["content"]
      if lContent.kind == JArray:
        for it in lContent.elems:
          outResp.logprobs.add(LogprobItem(token: it{"token"}.getStr(""), logprob: it{"logprob"}.getFloat(-99.0), textOffset: -1))
  if parsed.hasKey("usage") and parsed["usage"].kind == JObject:
    let u = parsed["usage"]
    outResp.promptTokens = u{"prompt_tokens"}.getInt(0)
    outResp.completionTokens = u{"completion_tokens"}.getInt(0)
    outResp.totalTokens = u{"total_tokens"}.getInt(0)
  if outResp.totalTokens == 0:
    outResp.promptTokens = max(1, canonical(messages).len div 4)
    outResp.completionTokens = max(1, outResp.content.len div 4)
    outResp.totalTokens = outResp.promptTokens + outResp.completionTokens
  return outResp

proc callTextCompletionsAsync(prompt: string, maxTokens: int, temperature: float,
                              echo: bool, logprobs: int): Future[CompletionResponse] {.async.} =
  let key = apiKey()
  if key.len == 0: raise newException(IOError, "MODULAR_API_KEY not configured")
  if prompt.len > MaxPromptBytes:
    raise newException(ValueError, "prompt exceeds byte bound: " & $prompt.len & " > " & $MaxPromptBytes)
  var body = %*{
    "model": ModularModel,
    "prompt": prompt,
    "max_tokens": max(1, min(maxTokens, MaxTokens)),
    "temperature": temperature,
    "top_p": TopP,
    "frequency_penalty": FrequencyPenalty,
    "presence_penalty": PresencePenalty,
    "seed": Seed,
    "echo": echo,
    "logprobs": max(0, min(logprobs, 20))
  }
  var client = newAsyncHttpClient(maxRedirects = 0)
  defer: client.close()
  client.headers = newHttpHeaders({
    "Authorization": "Bearer " & key,
    "Content-Type": "application/json",
    "Accept": "application/json"
  })
  let requestFuture = client.request(ModularBaseUrl & "/completions", httpMethod = HttpPost, body = $body)
  let resp = await awaitBounded(requestFuture, HttpTimeoutMs, "completion request")
  let bodyFuture = resp.body()
  let raw = await awaitBounded(bodyFuture, HttpTimeoutMs, "completion response body")
  if resp.code.int < 200 or resp.code.int >= 300:
    raise newException(IOError, "upstream completion status " & $resp.code.int & ": " & (if raw.len > 4096: raw[0 ..< 4096] else: raw))
  let parsed = parseJson(raw)
  var outResp = CompletionResponse()
  if parsed.hasKey("choices") and parsed["choices"].kind == JArray and parsed["choices"].elems.len > 0:
    let ch = parsed["choices"][0]
    outResp.text = ch{"text"}.getStr("")
    if ch.hasKey("logprobs") and ch["logprobs"].kind == JObject:
      let lp = ch["logprobs"]
      let toks = if lp.hasKey("tokens"): lp["tokens"] else: newJArray()
      let vals = if lp.hasKey("token_logprobs"): lp["token_logprobs"] else: newJArray()
      let offs = if lp.hasKey("text_offset"): lp["text_offset"] else: newJArray()
      if toks.kind == JArray:
        for i in 0 ..< toks.elems.len:
          let value = if vals.kind == JArray and i < vals.elems.len and vals[i].kind in {JInt, JFloat}: vals[i].getFloat() else: -99.0
          let offset = if offs.kind == JArray and i < offs.elems.len and offs[i].kind == JInt: offs[i].getInt(-1) else: -1
          outResp.logprobs.add(LogprobItem(token: toks[i].getStr(""), logprob: value, textOffset: offset))
  if parsed.hasKey("usage") and parsed["usage"].kind == JObject:
    let u = parsed["usage"]
    outResp.promptTokens = u{"prompt_tokens"}.getInt(0)
    outResp.completionTokens = u{"completion_tokens"}.getInt(0)
    outResp.totalTokens = u{"total_tokens"}.getInt(0)
  if outResp.totalTokens == 0:
    outResp.promptTokens = max(1, prompt.len div 4)
    outResp.completionTokens = max(1, outResp.text.len div 4)
    outResp.totalTokens = outResp.promptTokens + outResp.completionTokens
  return outResp

proc chargeTokens(tenantId, taskId: string, used: int): bool =
  if used <= 0: return true
  acquire(store.lock)
  defer: release(store.lock)
  execRawUnlocked(store, "BEGIN IMMEDIATE;")
  try:
    discard execUnlocked(store,
      "UPDATE tenants SET tokens_used=tokens_used+? WHERE tenant_id=? AND tokens_used+?<=token_budget",
      @[%used, %tenantId, %used])
    if sqlite3_changes(store.handle) != 1:
      execRawUnlocked(store, "ROLLBACK;")
      return false
    if taskId.len > 0:
      discard execUnlocked(store, "UPDATE tasks SET tokens_used=tokens_used+?, updated_at=? WHERE task_id=? AND tenant_id=?",
                           @[%used, %nowF(), %taskId, %tenantId])
      if sqlite3_changes(store.handle) != 1:
        execRawUnlocked(store, "ROLLBACK;")
        return false
    execRawUnlocked(store, "COMMIT;")
    return true
  except CatchableError:
    try: execRawUnlocked(store, "ROLLBACK;")
    except CatchableError: discard
    raise

proc extractJsonObject(s: string): JsonNode =
  var i = 0
  while i < s.len:
    if s[i] == '{':
      var depth = 0
      var inStr = false
      var esc = false
      var j = i
      while j < s.len:
        let c = s[j]
        if inStr:
          if esc: esc = false
          elif c == '\\': esc = true
          elif c == '"': inStr = false
        else:
          if c == '"': inStr = true
          elif c == '{': inc depth
          elif c == '}':
            dec depth
            if depth == 0:
              let cand = s[i .. j]
              try:
                let p = parseJson(cand)
                if p.kind == JObject: return p
              except CatchableError: discard
              break
        inc j
    inc i
  return nil

type
  TaskHandle = ref object
    taskId: string
    tenantId: string
    title: string
    spec: JsonNode
    sigma: JsonNode
    obs: JsonNode
    stepIndex: int
    maxSteps: int
    status: string
    stopRequested: bool
    paused: bool
    loopActive: Atomic[bool]
    transitionBusy: bool
    lock: Lock
    subscribers: seq[proc(ev: JsonNode) {.closure, gcsafe.}]
    cognition: JsonNode
    cognitionAt: float
    allowedTools: HashSet[string]
    verified: bool
    terminalReason: string
    broadcastAttached: bool

var
  activeTasks = initTable[string, TaskHandle]()
  tasksLock: Lock
  skillGateLock: Lock
  skillGateBusy = false

proc emit(h: TaskHandle, ev: JsonNode) =
  acquire(h.lock)
  let subs = h.subscribers
  release(h.lock)
  for s in subs:
    try: s(ev)
    except CatchableError: discard

proc checkpoint(h: TaskHandle, action, patch, receipt: JsonNode) =
  acquire(h.lock)
  let stateCopy = copy(h.sigma)
  let obsCopy = copy(h.obs)
  let step = h.stepIndex
  let status = h.status
  let reason = h.terminalReason
  let verified = h.verified
  release(h.lock)
  let dig = digestOf(%*{"state": stateCopy, "obs": obsCopy, "step": step})
  store.execTransaction([
    SqlOperation(
      sql: "INSERT INTO checkpoints (task_id, tenant_id, step_index, state_json, obs_json, action_json, patch_json, receipt_json, digest, created_at) " &
           "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(task_id, step_index) DO UPDATE SET " &
           "state_json=excluded.state_json, obs_json=excluded.obs_json, action_json=excluded.action_json, " &
           "patch_json=excluded.patch_json, receipt_json=excluded.receipt_json, digest=excluded.digest, created_at=excluded.created_at",
      params: @[%h.taskId, %h.tenantId, %step, %($stateCopy), %($obsCopy), %($action), %($patch), %($receipt), %dig, %nowF()]),
    SqlOperation(
      sql: "UPDATE tasks SET step_index=?, state_json=?, latest_obs_json=?, status=?, terminal_reason=?, verified=?, updated_at=? WHERE task_id=? AND tenant_id=?",
      params: @[%step, %($stateCopy), %($obsCopy), %status, %reason, %(if verified: 1 else: 0), %nowF(), %h.taskId, %h.tenantId])
  ])

proc logRawTrace(h: TaskHandle, skillId: string, preState, action, obs, delta, postState, receipt: JsonNode, success: bool, latency: int) =
  acquire(h.lock)
  let step = h.stepIndex
  release(h.lock)
  let tid = newId("trc")
  let imHash = sha1Hex(h.taskId & ":" & $step & ":" & canonical(preState) & ":" & canonical(action) & ":" & canonical(obs) & ":" & canonical(postState))
  discard store.exec(
    "INSERT INTO raw_traces (trace_id, task_id, tenant_id, step_index, initial_state_json, skill_id, " &
    "action_json, obs_json, delta_json, post_state_json, success, latency_ms, receipt_json, immutable_hash, created_at) " &
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    @[%tid, %h.taskId, %h.tenantId, %step, %($preState), %skillId,
      %($action), %($obs), %($delta), %($postState), %(if success: 1 else: 0),
      %latency, %($receipt), %imHash, %nowF()])

proc decodePointerToken(s: string): string =
  result = ""
  var i = 0
  while i < s.len:
    if s[i] == '~' and i + 1 < s.len:
      if s[i + 1] == '0':
        result.add('~')
        i += 2
        continue
      if s[i + 1] == '1':
        result.add('/')
        i += 2
        continue
    result.add(s[i])
    inc i

proc jsonPointerGet(root: JsonNode, pointer: string): Option[JsonNode] =
  if pointer.len == 0: return some(root)
  if not pointer.startsWith("/"): return none(JsonNode)
  var cur = root
  let rawParts = pointer[1 .. ^1].split('/')
  for raw in rawParts:
    let part = decodePointerToken(raw)
    if cur == nil: return none(JsonNode)
    case cur.kind
    of JObject:
      if not cur.hasKey(part): return none(JsonNode)
      cur = cur[part]
    of JArray:
      var idx = -1
      try: idx = parseInt(part)
      except CatchableError: return none(JsonNode)
      if idx < 0 or idx >= cur.elems.len: return none(JsonNode)
      cur = cur[idx]
    else:
      return none(JsonNode)
  some(cur)

proc verifyTerminal(h: TaskHandle): (bool, JsonNode) =
  acquire(h.lock)
  let sigma = copy(h.sigma)
  let spec = copy(h.spec)
  release(h.lock)
  var rep = newJArray()
  var allOk = true
  var configured = false
  if spec.hasKey("verifiers") and spec["verifiers"].kind == JArray:
    for verifier in spec["verifiers"].elems:
      if verifier.kind != JObject:
        allOk = false
        rep.add(%*{"verifier": "invalid", "ok": false, "error": "verifier must be an object"})
        continue
      configured = true
      let kind = verifier{"type"}.getStr(verifier{"verifier"}.getStr(""))
      case kind
      of "state_path_equals":
        let path = verifier{"path"}.getStr("")
        let actualOpt = jsonPointerGet(sigma, path)
        let expected = if verifier.hasKey("expected"): verifier["expected"] elif verifier.hasKey("value"): verifier["value"] else: newJNull()
        let ok = actualOpt.isSome and canonical(actualOpt.get()) == canonical(expected)
        if not ok: allOk = false
        rep.add(%*{"verifier": kind, "path": path, "ok": ok, "expected": expected,
                   "actual": (if actualOpt.isSome: actualOpt.get() else: newJNull())})
      of "file_exists":
        let rel = verifier{"path"}.getStr("")
        var ok = false
        try: ok = rel.len > 0 and fileExists(safeJoin(h.tenantId, rel))
        except CatchableError: ok = false
        if not ok: allOk = false
        rep.add(%*{"verifier": kind, "path": rel, "ok": ok})
      of "file_contains":
        let rel = verifier{"path"}.getStr("")
        let needle = verifier{"needle"}.getStr(verifier{"text"}.getStr(""))
        var ok = false
        try:
          let full = safeJoin(h.tenantId, rel)
          ok = fileExists(full) and needle.len > 0 and needle in readFile(full)
        except CatchableError:
          ok = false
        if not ok: allOk = false
        rep.add(%*{"verifier": kind, "path": rel, "needle": needle, "ok": ok})
      of "all_subgoals_resolved":
        var unresolved = 0
        if sigma.hasKey("subgoals") and sigma["subgoals"].kind == JArray:
          for it in sigma["subgoals"].elems:
            if it.kind == JObject and it{"status"}.getStr("open") in ["open", "in_progress"]: inc unresolved
        let ok = unresolved == 0
        if not ok: allOk = false
        rep.add(%*{"verifier": kind, "ok": ok, "unresolved": unresolved})
      of "no_blockers":
        let count = if sigma.hasKey("blockers") and sigma["blockers"].kind == JArray: sigma["blockers"].elems.len else: 0
        let ok = count == 0
        if not ok: allOk = false
        rep.add(%*{"verifier": kind, "ok": ok, "blockers_count": count})
      else:
        allOk = false
        rep.add(%*{"verifier": kind, "ok": false, "error": "unknown verifier"})
  if not configured:
    let p = sigma{"progress"}.getFloat(0.0)
    let pOk = p >= 0.999
    if not pOk: allOk = false
    rep.add(%*{"verifier": "state_path_equals", "path": "/progress", "ok": pOk, "expected": 1.0, "actual": p})
    var unresolved = 0
    if sigma.hasKey("subgoals") and sigma["subgoals"].kind == JArray:
      for it in sigma["subgoals"].elems:
        if it.kind == JObject and it{"status"}.getStr("open") in ["open", "in_progress"]: inc unresolved
    let subOk = unresolved == 0
    if not subOk: allOk = false
    rep.add(%*{"verifier": "all_subgoals_resolved", "ok": subOk, "unresolved": unresolved})
    let blockerCount = if sigma.hasKey("blockers") and sigma["blockers"].kind == JArray: sigma["blockers"].elems.len else: 0
    let blockerOk = blockerCount == 0
    if not blockerOk: allOk = false
    rep.add(%*{"verifier": "no_blockers", "ok": blockerOk, "blockers_count": blockerCount})
    if spec.hasKey("required_files") and spec["required_files"].kind == JArray:
      for rf in spec["required_files"].elems:
        let rel = rf.getStr()
        var ok = false
        try: ok = fileExists(safeJoin(h.tenantId, rel))
        except CatchableError: ok = false
        if not ok: allOk = false
        rep.add(%*{"verifier": "file_exists", "ok": ok, "path": rel})
  (allOk, rep)

proc boundUtf8Bytes(s: string, maxBytes: int): string =
  if maxBytes <= 0: return ""
  if s.len <= maxBytes: return s
  var n = maxBytes
  while n > 0 and n < s.len and (s[n].ord and 0xc0) == 0x80: dec n
  if n <= 0: return ""
  s[0 ..< n]

proc haltForBudget(h: TaskHandle) =
  acquire(h.lock)
  h.status = "halted"
  h.terminalReason = "budget_exhausted"
  h.stopRequested = true
  h.obs = %*{"error": "budget_exhausted", "step": h.stepIndex}
  release(h.lock)
  h.checkpoint(newJObject(), newJObject(), %*{"error": "budget_exhausted"})
  h.emit(%*{"type": "budget_exhausted", "task_id": h.taskId})

proc updatePolicyWeight(tenantId, token: string, delta: float) =
  let key = token.toLowerAscii().strip()
  if key.len < 2: return
  let boundedDelta = clamp(delta, -1.0, 1.0) * 0.1
  discard store.exec(
    "INSERT INTO policy_weights (tenant_id, token, weight, updates, updated_at) VALUES (?,?,?,1,?) " &
    "ON CONFLICT(tenant_id, token) DO UPDATE SET " &
    "weight=MIN(4.0, MAX(-4.0, policy_weights.weight + ?)), updates=policy_weights.updates+1, updated_at=?",
    @[%tenantId, %key, %boundedDelta, %nowF(), %boundedDelta, %nowF()])

proc runTokenLevelDistillation(h: TaskHandle, reflectionPatch: JsonNode) {.async.} =
  acquire(h.lock)
  let specSnapshot = copy(h.spec)
  let stateSnapshot = copy(h.sigma)
  release(h.lock)
  let cleanContext = "SPEC:\n" & boundUtf8Bytes(canonical(specSnapshot), 49152) & "\nSTATE:\n" & boundUtf8Bytes(canonical(stateSnapshot), 131072)
  let commonSuffix = "\nOUTPUT EXACTLY ONE JSON OBJECT WITH delta_sigma, action, AND terminal.\nTARGET_JSON:\n"
  let teacherPrefix = "You are the privileged teacher policy. Use the reflection patch to produce the optimal corrected transition.\n" &
                      cleanContext & "\nPRIVILEGED_REFLECTION:\n" & boundUtf8Bytes(canonical(reflectionPatch), 32768) & commonSuffix
  let studentPrefix = "You are the student policy. Produce the optimal transition from the clean task context.\n" & cleanContext & commonSuffix
  try:
    let teacher = await callTextCompletionsAsync(teacherPrefix, 2048, 0.2, false, 20)
    if not chargeTokens(h.tenantId, h.taskId, teacher.totalTokens):
      haltForBudget(h)
      return
    let targetNode = extractJsonObject(teacher.text)
    if targetNode == nil or teacher.logprobs.len == 0:
      h.emit(%*{"type": "distillation_skipped", "task_id": h.taskId, "reason": "teacher_target_invalid"})
      return
    let targetText = teacher.text
    if studentPrefix.len + targetText.len > MaxPromptBytes:
      h.emit(%*{"type": "distillation_skipped", "task_id": h.taskId, "reason": "aligned_target_exceeds_prompt_bound"})
      return
    let scoredPrompt = studentPrefix & targetText
    let student = await callTextCompletionsAsync(scoredPrompt, 1, 0.0, true, 20)
    if not chargeTokens(h.tenantId, h.taskId, student.totalTokens):
      haltForBudget(h)
      return
    var studentTarget: seq[LogprobItem] = @[]
    let targetStart = studentPrefix.len
    let targetEnd = scoredPrompt.len
    for item in student.logprobs:
      if item.textOffset >= targetStart and item.textOffset < targetEnd:
        studentTarget.add(item)
    if studentTarget.len != teacher.logprobs.len:
      h.emit(%*{"type": "distillation_skipped", "task_id": h.taskId, "reason": "token_alignment_length_mismatch",
                "teacher_tokens": teacher.logprobs.len, "student_tokens": studentTarget.len})
      return
    for i in 0 ..< teacher.logprobs.len:
      if teacher.logprobs[i].token != studentTarget[i].token:
        h.emit(%*{"type": "distillation_skipped", "task_id": h.taskId, "reason": "token_alignment_identity_mismatch", "index": i})
        return
    var relevant = initHashSet[string]()
    if targetNode.hasKey("action"):
      for term in contentTerms(canonical(targetNode["action"])): relevant.incl(term)
      let toolName = targetNode["action"]{"tool"}.getStr("").toLowerAscii()
      if toolName.len > 0:
        relevant.incl(toolName)
        for part in toolName.split('_'):
          if part.len >= 2: relevant.incl(part)
    var rkl = 0.0
    var meanAdv = 0.0
    var matched = 0
    for i in 0 ..< teacher.logprobs.len:
      let tLog = clamp(teacher.logprobs[i].logprob, -60.0, 0.0)
      let sLog = clamp(studentTarget[i].logprob, -60.0, 0.0)
      let tProb = exp(tLog)
      let sProb = exp(sLog)
      let adv = tProb - sProb
      rkl += sProb * (sLog - tLog)
      meanAdv += adv
      let tokenTerms = contentTerms(teacher.logprobs[i].token)
      for term in tokenTerms:
        if term in relevant:
          updatePolicyWeight(h.tenantId, term, adv)
          inc matched
    if teacher.logprobs.len > 0: meanAdv /= teacher.logprobs.len.float
    if matched == 0:
      for term in relevant: updatePolicyWeight(h.tenantId, term, meanAdv)
    h.emit(%*{"type": "distillation", "task_id": h.taskId, "aligned_tokens": teacher.logprobs.len,
              "reverse_kl": rkl, "mean_advantage": meanAdv, "policy_updates": max(matched, relevant.len * (if matched == 0: 1 else: 0))})
  except CatchableError as e:
    h.emit(%*{"type": "distillation_error", "task_id": h.taskId, "error": e.msg})

proc reflectAndDistill(h: TaskHandle, verifierReport: JsonNode) {.async.} =
  let rows = store.query("SELECT step_index, action_json, obs_json, success FROM raw_traces WHERE task_id=? ORDER BY step_index DESC LIMIT 10", @[%h.taskId])
  var digestArr = newJArray()
  for r in rows:
    digestArr.add(%*{"step": getInt(r, "step_index"), "action": getJson(r, "action_json"),
                     "obs": getJson(r, "obs_json"), "success": getInt(r, "success") == 1})
  acquire(h.lock)
  let stateSnapshot = copy(h.sigma)
  release(h.lock)
  let prompt = %*[
    {"role": "system", "content":
      "You are a meta-reflection engine. Diagnose failure, attribute memory component, suggest pivot action and skill patch. " &
      "Output JSON with failure_point, root_cause, pivot_action, attribution, and skill_patch."},
    {"role": "user", "content":
      "TERMINAL STATE:\n" & boundUtf8Bytes(canonical(stateSnapshot), 131072) & "\nVERIFIER REPORT:\n" & boundUtf8Bytes(canonical(verifierReport), 32768) &
      "\nTRACE DIGEST:\n" & boundUtf8Bytes(canonical(digestArr), 49152)}
  ]
  var patchNode = newJObject()
  try:
    let resp = await callChatCompletionsAsync(prompt, 4096, 0.2, true, false)
    if not chargeTokens(h.tenantId, h.taskId, resp.totalTokens):
      haltForBudget(h)
      return
    let parsed = extractJsonObject(resp.content)
    if parsed != nil: patchNode = parsed
  except CatchableError as e:
    patchNode = %*{"failure_point": "reflection_failed", "root_cause": e.msg, "pivot_action": "retry", "attribution": "environment"}
  let rid = newId("ref")
  discard store.exec(
    "INSERT INTO reflections (reflection_id, task_id, tenant_id, patch_json, failure_point, pivot_action, attribution, verifier_report_json, created_at) " &
    "VALUES (?,?,?,?,?,?,?,?,?)",
    @[%rid, %h.taskId, %h.tenantId, %($patchNode), %patchNode{"failure_point"}.getStr(""),
      %patchNode{"pivot_action"}.getStr(""), %patchNode{"attribution"}.getStr("environment"),
      %($verifierReport), %nowF()])
  h.emit(%*{"type": "reflection", "task_id": h.taskId, "patch": patchNode})
  await runTokenLevelDistillation(h, patchNode)

proc runRegressionGate(tenantId: string): Future[JsonNode] {.async.} =
  var tests = newJArray()
  var passed = 0
  var total = 0
  let testDir = ".diagnostics/" & newId("gate")
  let testPath = testDir & "/sandbox_reg_test.txt"
  var fPass = false
  try:
    let writeRes = await toolRegistry["write_file"].handler(tenantId, %*{"path": testPath, "content": "alpha\nbeta\n"})
    let appRes = await toolRegistry["append_file"].handler(tenantId, %*{"path": testPath, "lines": ["beta", "gamma"], "unique": true})
    let repRes = await toolRegistry["replace_lines"].handler(tenantId, %*{"path": testPath, "start_line": 2, "end_line": 2, "lines": ["delta"]})
    let chkRes = await toolRegistry["check_lines"].handler(tenantId, %*{"path": testPath, "lines": ["alpha", "delta", "gamma"]})
    var traversalBlocked = false
    try: discard safeJoin(tenantId, "../tenant_escape_probe")
    except ValueError: traversalBlocked = true
    fPass = writeRes.ok and appRes.ok and appRes.payload{"added_count"}.getInt(0) == 1 and
            repRes.ok and chkRes.ok and chkRes.payload{"missing_count"}.getInt(1) == 0 and traversalBlocked
  except CatchableError:
    fPass = false
  inc total
  if fPass: inc passed
  tests.add(%*{"name": "filesystem_sandbox", "passed": fPass})
  try:
    let full = safeJoin(tenantId, testPath)
    if fileExists(full): removeFile(full)
    let dirFull = safeJoin(tenantId, testDir)
    if dirExists(dirFull): removeDir(dirFull)
  except CatchableError: discard

  let (mOk1, mVal1, _) = evalMathExpression("((17 * 23 + sqrt(144)) / 5) - ln(exp(2))")
  let (mOk2, _, _) = evalMathExpression("100 / 0")
  let (mOk3, mVal3, _) = evalMathExpression("-2^2 + 2^3^2")
  let mPass = mOk1 and abs(mVal1 - 78.6) < 1e-9 and (not mOk2) and mOk3 and abs(mVal3 - 508.0) < 1e-9
  inc total
  if mPass: inc passed
  tests.add(%*{"name": "math_eval_boundary", "passed": mPass})

  let baseS = defaultSigma("test")
  for i in 0 ..< 180: baseS["facts"]["f" & $i] = %i
  baseS["facts"]["delete_me"] = %"x"
  let patchS = %*{"facts": {"delete_me": newJNull(), "f_final": "v2"}, "progress": 0.5}
  let (patchOk, _) = validatePatch(patchS)
  let mergedS = deepMerge(baseS, patchS)
  pruneSigma(mergedS)
  let (stateOk, _) = validateSigma(mergedS)
  let (forbiddenOk, _) = validatePatch(%*{"nested": {"messages": []}})
  let sHits = searchSkills(tenantId, "filesystem write lines", 3)
  let activeSkills = store.query("SELECT COUNT(*) AS n FROM skills WHERE tenant_id=? AND active=1", @[%tenantId])
  let retrievalOk = activeSkills.len == 0 or getInt(activeSkills[0], "n", 0) == 0 or sHits.len > 0
  let sPass = patchOk and stateOk and mergedS{"progress"}.getFloat(0.0) == 0.5 and
              (not mergedS{"facts"}.hasKey("delete_me")) and mergedS{"facts"}.fields.len <= 128 and
              (not forbiddenOk) and countNodes(mergedS) <= 4000 and canonical(mergedS).len <= MaxStateBytes and retrievalOk
  inc total
  if sPass: inc passed
  tests.add(%*{"name": "memory_routing_state_pruning", "passed": sPass})

  let diagnosticRows = store.query("SELECT COUNT(*) AS n FROM diagnostics WHERE tenant_id=?", @[%tenantId])
  let rolloutAvailable = diagnosticRows.len > 0 and getInt(diagnosticRows[0], "n", 0) > 0
  tests.add(%*{"name": "optional_micro_rollout", "passed": true, "executed": false, "available": rolloutAvailable})
  let score = if total > 0: passed.float / total.float else: 1.0
  return %*{"total": total, "passed": passed, "score": score, "tests": tests, "all_passed": passed == total}

proc finalizeTask(h: TaskHandle) {.async.} =
  let (ok, report) = verifyTerminal(h)
  acquire(h.lock)
  h.verified = ok
  let previousStatus = h.status
  if previousStatus notin ["halted"]:
    h.status = if ok: "succeeded" else: "failed"
  if h.terminalReason.len == 0:
    h.terminalReason = if ok: "all external verifiers satisfied" else: "external verifiers failed"
  var obs = copy(h.obs)
  obs["verification_report"] = report
  h.obs = obs
  let shouldReflect = not ok and previousStatus != "halted"
  release(h.lock)
  h.checkpoint(newJObject(), newJObject(), report)
  h.emit(%*{"type": "verification", "task_id": h.taskId, "verified": ok, "report": report, "status": h.status})
  if shouldReflect:
    await reflectAndDistill(h, report)
  discard store.exec("UPDATE tasks SET status=?, verified=?, terminal_reason=?, updated_at=? WHERE task_id=?",
                     @[%h.status, %(if h.verified: 1 else: 0), %h.terminalReason, %nowF(), %h.taskId])
  h.emit(%*{"type": "done", "task_id": h.taskId, "status": h.status, "verified": h.verified, "reason": h.terminalReason})

proc system2Think(h: TaskHandle) {.async.} =
  acquire(h.lock)
  if h.status != "running" or h.paused or h.stopRequested or h.transitionBusy:
    release(h.lock)
    return
  let sigmaSnapshot = copy(h.sigma)
  let obsSnapshot = copy(h.obs)
  let specSnapshot = copy(h.spec)
  let stepSnapshot = h.stepIndex
  let stateDigest = digestOf(sigmaSnapshot)
  release(h.lock)
  let queryText = sigmaSnapshot{"goal"}.getStr("") & " " & sigmaSnapshot{"step_summary"}.getStr("")
  let skills = searchSkills(h.tenantId, queryText, 3)
  var skillArr = newJArray()
  for sk in skills:
    skillArr.add(%*{"name": getStr(sk, "name"), "trigger": getStr(sk, "trigger_spec"), "procedure": getStr(sk, "procedure_spec")})
  let messages = %*[
    {"role": "system", "content":
      "You are System 2, the deliberative planning engine. Output strict JSON with subgoal, strategy, gate in [0,1], " &
      "cognition as exactly 16 numbers in [-1,1], and queued_micro_actions as immediate authorized tool calls."},
    {"role": "user", "content":
      "SPEC:\n" & boundUtf8Bytes(canonical(specSnapshot), 49152) & "\nSTATE:\n" & boundUtf8Bytes(canonical(sigmaSnapshot), 131072) &
      "\nOBS:\n" & boundUtf8Bytes(canonical(obsSnapshot), 49152) & "\nACTIVE SKILLS:\n" & boundUtf8Bytes(canonical(skillArr), 16384)}
  ]
  try:
    let resp = await callChatCompletionsAsync(messages, 2048, 0.4, true, false)
    if not chargeTokens(h.tenantId, h.taskId, resp.totalTokens):
      haltForBudget(h)
      return
    let node = extractJsonObject(resp.content)
    if node == nil: return
    var vec = newJArray()
    if node.hasKey("cognition") and node["cognition"].kind == JArray:
      for it in node["cognition"].elems:
        if vec.len >= 16: break
        vec.add(%clamp(it.getFloat(0.0), -1.0, 1.0))
    while vec.len < 16: vec.add(%0.0)
    let gate = clamp(node{"gate"}.getFloat(0.5), 0.0, 1.0)
    let generatedAt = nowF()
    let cog = %*{"subgoal": node{"subgoal"}.getStr(""), "strategy": node{"strategy"}.getStr(""),
                 "gate": gate, "vector": vec, "generated_at": generatedAt}
    var validatedActions = newJArray()
    if node.hasKey("queued_micro_actions") and node["queued_micro_actions"].kind == JArray:
      for act in node["queued_micro_actions"].elems:
        if validatedActions.elems.len >= 16: break
        if act.kind != JObject: continue
        let toolName = act{"tool"}.getStr("")
        if toolName.len == 0 or not toolRegistry.hasKey(toolName) or toolName notin h.allowedTools: continue
        var argsNode = act{"args"}
        if argsNode == nil or argsNode.kind != JObject: argsNode = newJObject()
        validatedActions.add(%*{"tool": toolName, "args": argsNode})
    acquire(h.lock)
    let stale = h.status != "running" or h.paused or h.stopRequested or h.transitionBusy or
                h.stepIndex != stepSnapshot or digestOf(h.sigma) != stateDigest
    if stale:
      release(h.lock)
      h.emit(%*{"type": "cognition_stale", "task_id": h.taskId, "step": stepSnapshot})
      return
    h.cognition = cog
    h.cognitionAt = generatedAt
    if not h.sigma.hasKey("system1") or h.sigma["system1"].kind != JObject: h.sigma["system1"] = newJObject()
    var q = if h.sigma["system1"].hasKey("queued_actions") and h.sigma["system1"]["queued_actions"].kind == JArray:
              h.sigma["system1"]["queued_actions"] else: newJArray()
    for act in validatedActions.elems:
      if q.elems.len >= 32: break
      q.add(act)
    h.sigma["system1"]["queued_actions"] = q
    release(h.lock)
    let cid = newId("cog")
    discard store.exec(
      "INSERT INTO cognition (cog_id, task_id, tenant_id, step_index, vector_json, gate, subgoal, strategy, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
      @[%cid, %h.taskId, %h.tenantId, %stepSnapshot, %($vec), %gate, %node{"subgoal"}.getStr(""), %node{"strategy"}.getStr(""), %generatedAt])
    h.emit(%*{"type": "cognition", "task_id": h.taskId, "cognition": cog, "queued_micro_actions": validatedActions.elems.len})
  except CatchableError as e:
    h.emit(%*{"type": "cognition_error", "task_id": h.taskId, "error": e.msg})

proc tryBeginTransition(h: TaskHandle): bool =
  acquire(h.lock)
  if h.transitionBusy or h.status != "running" or h.paused or h.stopRequested or h.stepIndex >= h.maxSteps:
    release(h.lock)
    return false
  h.transitionBusy = true
  release(h.lock)
  true

proc endTransition(h: TaskHandle) =
  acquire(h.lock)
  h.transitionBusy = false
  release(h.lock)

proc executeStep(h: TaskHandle) {.async.} =
  acquire(h.lock)
  let preSigma = copy(h.sigma)
  let baseObs = copy(h.obs)
  let specSnapshot = copy(h.spec)
  let step = h.stepIndex
  let cogBase = if h.cognition != nil: copy(h.cognition) else: newJObject()
  let cogAge = if h.cognitionAt > 0.0: max(0.0, nowF() - h.cognitionAt) else: 0.0
  release(h.lock)
  let qText = preSigma{"goal"}.getStr("") & " " & preSigma{"step_summary"}.getStr("")
  let skills = searchSkills(h.tenantId, qText, 2)
  let wiki = searchKnowledge(h.tenantId, qText, 2)
  var cog = cogBase
  if cog.kind == JObject and cog.len > 0:
    cog["staleness_encoding"] = stalenessEncoding(cogAge)
    cog["staleness_seconds"] = %cogAge
  var attempt = 0
  var committed = false
  var lastErrReport = newJObject()
  while attempt < MaxRetryPerStep and not committed:
    inc attempt
    var curObs = copy(baseObs)
    if attempt > 1:
      curObs["retry_attempt"] = %attempt
      curObs["validation_error"] = lastErrReport
    let pText = "IMMUTABLE TASK SPECIFICATION:\n" & boundUtf8Bytes(canonical(specSnapshot), 49152) &
                "\n\nALLOWED TOOLS:\n" & boundUtf8Bytes(canonical(toolCatalog(h.allowedTools)), 24576) &
                "\n\nOUTPUT CONTRACT: Emit exactly one JSON object with reasoning, delta_sigma, action containing tool and args, and terminal boolean. " &
                "Reasoning is discarded. Never include history, transcript, messages, chain_of_thought, or scratchpad in delta_sigma."
    var uText = "STATE SIGMA:\n" & boundUtf8Bytes(canonical(preSigma), 131072) & "\n\nOBSERVATION:\n" & boundUtf8Bytes(canonical(curObs), 32768)
    if skills.len > 0:
      var sArr = newJArray()
      for sk in skills: sArr.add(%*{"name": getStr(sk, "name"), "trigger": getStr(sk, "trigger_spec"), "procedure": getStr(sk, "procedure_spec")})
      uText.add("\n\nEXPERIENTIAL SKILLS:\n" & boundUtf8Bytes(canonical(sArr), 16384))
    if wiki.len > 0:
      var wArr = newJArray()
      for w in wiki: wArr.add(%*{"slug": getStr(w, "slug"), "category": getStr(w, "category"), "excerpt": getStr(w, "excerpt")})
      uText.add("\n\nKNOWLEDGE PLAYBOOK:\n" & boundUtf8Bytes(canonical(wArr), 16384))
    if cog.kind == JObject and cog.len > 0:
      uText.add("\n\nSYSTEM 2 COGNITION:\n" & boundUtf8Bytes(canonical(cog), 8192))
    let messages = %*[{"role": "system", "content": pText}, {"role": "user", "content": uText}]
    var resp: LlmResponse
    try:
      resp = await callChatCompletionsAsync(messages, 8192, Temperature, true, false)
    except CatchableError as e:
      lastErrReport = %*{"error": e.msg, "stage": "llm_call"}
      if attempt < MaxRetryPerStep: await sleepAsync(250)
      continue
    if not chargeTokens(h.tenantId, h.taskId, resp.totalTokens):
      haltForBudget(h)
      return
    let parsed = extractJsonObject(resp.content)
    if parsed == nil:
      lastErrReport = %*{"error": "no json object returned", "stage": "json_parse"}
      continue
    let delta = if parsed.hasKey("delta_sigma") and parsed["delta_sigma"].kind == JObject: parsed["delta_sigma"] else: newJObject()
    let (patchOk, patchErrs) = validatePatch(delta)
    if not patchOk:
      lastErrReport = %*{"error": patchErrs.join("; "), "stage": "patch_validation"}
      continue
    let action = if parsed.hasKey("action") and parsed["action"].kind == JObject: parsed["action"] else: %*{"tool": "none", "args": {}}
    let termReq = parsed{"terminal"}.getBool(false)
    let candSigma = deepMerge(preSigma, delta)
    pruneSigma(candSigma)
    let (vOk, vErrs) = validateSigma(candSigma)
    if not vOk:
      lastErrReport = %*{"error": vErrs.join("; "), "stage": "state_validation"}
      continue
    let tName = action{"tool"}.getStr("none")
    var toolRes = ToolResult(ok: true, payload: newJObject(), receipt: "none", message: "no-op")
    let startTime = getMonoTime()
    if tName notin ["none", "finish"]:
      if not toolRegistry.hasKey(tName):
        toolRes = ToolResult(ok: false, payload: newJObject(), receipt: "unknown", message: "unknown tool: " & tName)
      elif tName notin h.allowedTools:
        toolRes = ToolResult(ok: false, payload: newJObject(), receipt: "forbidden", message: "tool unauthorized")
      else:
        try:
          var tArgs = action{"args"}
          if tArgs == nil or tArgs.kind != JObject: tArgs = newJObject()
          toolRes = await toolRegistry[tName].handler(h.tenantId, tArgs)
        except CatchableError as e:
          toolRes = ToolResult(ok: false, payload: newJObject(), receipt: "error", message: e.msg)
    let latency = int((getMonoTime() - startTime).inMilliseconds)
    if not toolRes.ok:
      var blockers = if candSigma.hasKey("blockers") and candSigma["blockers"].kind == JArray: candSigma["blockers"] else: newJArray()
      blockers.add(%*{"step": step, "reason": toolRes.message, "tool": tName, "at": nowF()})
      candSigma["blockers"] = blockers
      pruneSigma(candSigma)
    let (postOk, postErrs) = validateSigma(candSigma)
    if not postOk:
      lastErrReport = %*{"error": postErrs.join("; "), "stage": "post_tool_state_validation"}
      continue
    var nextObs = %*{"step": step, "tool": tName, "ok": toolRes.ok, "message": toolRes.message,
                     "payload": toolRes.payload, "receipt": toolRes.receipt, "latency_ms": latency}
    acquire(h.lock)
    if h.obs.hasKey("operator_message") and (not baseObs.hasKey("operator_message") or canonical(h.obs["operator_message"]) != canonical(baseObs{"operator_message"})):
      nextObs["operator_message"] = copy(h.obs["operator_message"])
    h.sigma = candSigma
    h.obs = nextObs
    if (termReq or tName == "finish") and h.status == "running": h.status = "verifying"
    release(h.lock)
    committed = true
    let skillUsed = if skills.len > 0: getStr(skills[0], "skill_id") else: ""
    h.logRawTrace(skillUsed, preSigma, action, nextObs, delta, candSigma, %*{"receipt": toolRes.receipt}, toolRes.ok, latency)
    h.checkpoint(action, delta, %*{"receipt": toolRes.receipt, "ok": toolRes.ok})
    h.emit(%*{"type": "step", "task_id": h.taskId, "step": step, "sigma": candSigma, "obs": nextObs, "action": action})
  if not committed:
    acquire(h.lock)
    h.status = "failed"
    h.terminalReason = "max retry attempts exhausted on validation error"
    h.stopRequested = true
    h.obs = lastErrReport
    release(h.lock)
    h.checkpoint(newJObject(), newJObject(), lastErrReport)

proc executeMicroAction(h: TaskHandle): Future[bool] {.async.} =
  if not tryBeginTransition(h): return false
  try:
    acquire(h.lock)
    if not h.sigma.hasKey("system1") or h.sigma["system1"].kind != JObject or
       not h.sigma["system1"].hasKey("queued_actions") or h.sigma["system1"]["queued_actions"].kind != JArray or
       h.sigma["system1"]["queued_actions"].elems.len == 0:
      release(h.lock)
      return false
    let preSigma = copy(h.sigma)
    let microAction = copy(h.sigma["system1"]["queued_actions"][0])
    h.sigma["system1"]["queued_actions"].delete(0)
    inc h.stepIndex
    let step = h.stepIndex
    release(h.lock)
    let tName = microAction{"tool"}.getStr("none")
    var tArgs = microAction{"args"}
    if tArgs == nil or tArgs.kind != JObject: tArgs = newJObject()
    let startTime = getMonoTime()
    var toolRes = ToolResult(ok: true, payload: newJObject(), receipt: "none", message: "no-op")
    if tName != "none":
      if not toolRegistry.hasKey(tName):
        toolRes = ToolResult(ok: false, payload: newJObject(), receipt: "unknown", message: "unknown tool: " & tName)
      elif tName notin h.allowedTools:
        toolRes = ToolResult(ok: false, payload: newJObject(), receipt: "forbidden", message: "tool unauthorized")
      else:
        try: toolRes = await toolRegistry[tName].handler(h.tenantId, tArgs)
        except CatchableError as e: toolRes = ToolResult(ok: false, payload: newJObject(), receipt: "error", message: e.msg)
    let latency = int((getMonoTime() - startTime).inMilliseconds)
    var obs = %*{"step": step, "tool": tName, "ok": toolRes.ok, "message": toolRes.message,
                 "payload": toolRes.payload, "receipt": toolRes.receipt, "latency_ms": latency, "micro": true}
    acquire(h.lock)
    if not toolRes.ok:
      var blockers = if h.sigma.hasKey("blockers") and h.sigma["blockers"].kind == JArray: h.sigma["blockers"] else: newJArray()
      blockers.add(%*{"step": step, "reason": toolRes.message, "tool": tName, "at": nowF()})
      h.sigma["blockers"] = blockers
      pruneSigma(h.sigma)
    h.obs = obs
    let postSigma = copy(h.sigma)
    release(h.lock)
    h.logRawTrace("", preSigma, microAction, obs, newJObject(), postSigma, %*{"receipt": toolRes.receipt}, toolRes.ok, latency)
    h.checkpoint(microAction, newJObject(), %*{"receipt": toolRes.receipt, "micro": true, "ok": toolRes.ok})
    h.emit(%*{"type": "step", "task_id": h.taskId, "step": step, "sigma": postSigma, "obs": obs, "micro": true})
    return true
  finally:
    endTransition(h)

proc system1Tick(h: TaskHandle) {.async.} =
  acquire(h.lock)
  let runnable = h.status == "running" and not h.stopRequested and not h.paused and h.stepIndex < h.maxSteps
  let hasMicro = runnable and h.sigma.hasKey("system1") and h.sigma["system1"].kind == JObject and
                 h.sigma["system1"].hasKey("queued_actions") and h.sigma["system1"]["queued_actions"].kind == JArray and
                 h.sigma["system1"]["queued_actions"].elems.len > 0
  let gateVal = if h.cognition != nil: h.cognition{"gate"}.getFloat(0.5) else: 0.5
  let stepIsZero = h.stepIndex == 0
  release(h.lock)
  if not runnable: return
  if hasMicro:
    discard await executeMicroAction(h)
    return
  if gateVal >= 0.5 or stepIsZero:
    if not tryBeginTransition(h): return
    try:
      acquire(h.lock)
      inc h.stepIndex
      release(h.lock)
      await executeStep(h)
    finally:
      endTransition(h)

proc system2Loop(h: TaskHandle) {.async.} =
  while true:
    await sleepAsync(System2HzInterval)
    acquire(h.lock)
    let keepRunning = not h.stopRequested and h.status in ["running", "queued"]
    let shouldThink = keepRunning and h.status == "running" and not h.paused and not h.transitionBusy
    release(h.lock)
    if not keepRunning: break
    if shouldThink: await h.system2Think()

proc system1Loop(h: TaskHandle) {.async.} =
  while true:
    await sleepAsync(System1HzInterval)
    acquire(h.lock)
    let keepRunning = not h.stopRequested and h.status in ["running", "queued"]
    let shouldTick = keepRunning and h.status == "running" and not h.paused
    let atLimit = h.stepIndex >= h.maxSteps
    release(h.lock)
    if not keepRunning: break
    if atLimit:
      acquire(h.lock)
      h.status = "verifying"
      if h.terminalReason.len == 0: h.terminalReason = "max_steps_reached"
      release(h.lock)
      break
    if shouldTick:
      await h.system1Tick()
      acquire(h.lock)
      let terminal = h.status in ["verifying", "failed", "halted"] or h.stopRequested
      release(h.lock)
      if terminal: break
  await h.finalizeTask()

proc taskSupervisor(h: TaskHandle) {.async.} =
  let s2 = system2Loop(h)
  try:
    await system1Loop(h)
  finally:
    acquire(h.lock)
    if h.status notin ["running", "queued"]: h.stopRequested = true
    release(h.lock)
    try: await s2
    except CatchableError: discard
    h.loopActive.store(false, moRelease)

proc launchTask(h: TaskHandle): bool =
  var expected = false
  if not h.loopActive.compareExchange(expected, true, moAcquireRelease, moAcquire): return false
  acquire(h.lock)
  h.status = "running"
  h.stopRequested = false
  h.paused = false
  release(h.lock)
  discard store.exec("UPDATE tasks SET status='running', updated_at=? WHERE task_id=?", @[%nowF(), %h.taskId])
  asyncCheck taskSupervisor(h)
  true

proc createTask(tenantRow: Row, title: string, spec: JsonNode, maxSteps: int): TaskHandle =
  let tid = getStr(tenantRow, "tenant_id")
  let taskId = newId("task")
  var nSpec = if spec != nil and spec.kind == JObject: copy(spec) else: newJObject()
  if not nSpec.hasKey("goal"): nSpec["goal"] = %title
  if canonical(nSpec).len > 65536: raise newException(ValueError, "task specification exceeds 65536 bytes")
  var sigma = if nSpec.hasKey("initial_state") and nSpec["initial_state"].kind == JObject: copy(nSpec["initial_state"]) else: defaultSigma(nSpec{"goal"}.getStr(title))
  if not sigma.hasKey("system1") or sigma["system1"].kind != JObject: sigma["system1"] = newJObject()
  if not sigma["system1"].hasKey("queued_actions") or sigma["system1"]["queued_actions"].kind != JArray: sigma["system1"]["queued_actions"] = newJArray()
  pruneSigma(sigma)
  let (stateOk, stateErrs) = validateSigma(sigma)
  if not stateOk: raise newException(ValueError, "invalid initial state: " & stateErrs.join("; "))
  let obs = %*{"step": 0, "status": "initialized", "message": "agent task launched"}
  let ts = nowF()
  let boundedMaxSteps = max(1, maxSteps)
  discard store.exec(
    "INSERT INTO tasks (task_id, tenant_id, title, spec_json, initial_state_json, state_json, " &
    "latest_obs_json, status, step_index, max_steps, tokens_used, terminal_reason, verified, created_at, updated_at) " &
    "VALUES (?,?,?,?,?,?,?,?,0,?,0,'',0,?,?)",
    @[%taskId, %tid, %title, %($nSpec), %($sigma), %($sigma), %($obs), %"queued", %boundedMaxSteps, %ts, %ts])
  var allowedSet = initHashSet[string]()
  let aj = getJson(tenantRow, "allowed_tools")
  if aj.kind == JArray:
    for it in aj.elems:
      let name = it.getStr()
      if toolRegistry.hasKey(name): allowedSet.incl(name)
  result = TaskHandle(taskId: taskId, tenantId: tid, title: title, spec: nSpec, sigma: sigma, obs: obs,
    stepIndex: 0, maxSteps: boundedMaxSteps, status: "queued", stopRequested: false, paused: false,
    transitionBusy: false, subscribers: @[], cognition: newJObject(), cognitionAt: 0.0,
    allowedTools: allowedSet, verified: false, terminalReason: "", broadcastAttached: false)
  initLock(result.lock)
  result.loopActive.store(false, moRelaxed)
  acquire(tasksLock)
  activeTasks[taskId] = result
  release(tasksLock)
  result.checkpoint(newJObject(), newJObject(), %*{"status": "initialized"})

proc restoreTask(taskId: string): TaskHandle =
  acquire(tasksLock)
  if activeTasks.hasKey(taskId):
    result = activeTasks[taskId]
    release(tasksLock)
    return result
  release(tasksLock)
  let rows = store.query("SELECT * FROM tasks WHERE task_id=?", @[%taskId])
  if rows.len == 0: return nil
  let r = rows[0]
  let tid = getStr(r, "tenant_id")
  let tRows = store.query("SELECT * FROM tenants WHERE tenant_id=?", @[%tid])
  if tRows.len == 0: return nil
  var allowedSet = initHashSet[string]()
  let aj = getJson(tRows[0], "allowed_tools")
  if aj.kind == JArray:
    for it in aj.elems:
      let name = it.getStr()
      if toolRegistry.hasKey(name): allowedSet.incl(name)
  var cog = newJObject()
  var cogAt = 0.0
  let cogRows = store.query("SELECT vector_json, gate, subgoal, strategy, created_at FROM cognition WHERE task_id=? ORDER BY created_at DESC LIMIT 1", @[%taskId])
  if cogRows.len > 0:
    var vec: JsonNode
    try: vec = parseJson(getStr(cogRows[0], "vector_json", "[]"))
    except CatchableError: vec = newJArray()
    cogAt = getFloat(cogRows[0], "created_at", 0.0)
    cog = %*{"vector": vec, "gate": getFloat(cogRows[0], "gate", 0.5), "subgoal": getStr(cogRows[0], "subgoal"),
             "strategy": getStr(cogRows[0], "strategy"), "generated_at": cogAt}
  result = TaskHandle(taskId: taskId, tenantId: tid, title: getStr(r, "title"), spec: getJson(r, "spec_json"),
    sigma: getJson(r, "state_json"), obs: getJson(r, "latest_obs_json"),
    stepIndex: getInt(r, "step_index").int, maxSteps: max(1, getInt(r, "max_steps", 1000).int),
    status: getStr(r, "status"), stopRequested: false, paused: false, transitionBusy: false, subscribers: @[],
    cognition: cog, cognitionAt: cogAt, allowedTools: allowedSet,
    verified: getInt(r, "verified") == 1, terminalReason: getStr(r, "terminal_reason"), broadcastAttached: false)
  initLock(result.lock)
  result.loopActive.store(false, moRelaxed)
  acquire(tasksLock)
  activeTasks[taskId] = result
  release(tasksLock)

proc attachBroadcast(h: TaskHandle)

proc resumePendingTasks() =
  let rows = store.query("SELECT task_id FROM tasks WHERE status IN ('queued', 'running')", @[])
  for r in rows:
    let h = restoreTask(getStr(r, "task_id"))
    if h != nil and h.stepIndex < h.maxSteps:
      attachBroadcast(h)
      discard launchTask(h)

type
  SseClient = ref object
    req: Request
    tenantId: string
    alive: bool
    queue: Deque[string]
    lock: Lock

var
  sseClients: seq[SseClient] = @[]
  sseLock: Lock

proc pushSse(c: SseClient, payload: string) =
  acquire(c.lock)
  if c.queue.len < 2048: c.queue.addLast(payload)
  release(c.lock)

proc broadcastTenant(tenantId: string, ev: JsonNode) =
  let payload = "data: " & canonical(ev) & "\n\n"
  acquire(sseLock)
  var live: seq[SseClient] = @[]
  for c in sseClients:
    if c.alive:
      if c.tenantId == tenantId: pushSse(c, payload)
      live.add(c)
  sseClients = live
  release(sseLock)

proc attachBroadcast(h: TaskHandle) =
  acquire(h.lock)
  if not h.broadcastAttached:
    let tid = h.tenantId
    h.subscribers.add(proc(ev: JsonNode) {.closure, gcsafe.} = broadcastTenant(tid, ev))
    h.broadcastAttached = true
  release(h.lock)

proc authenticate(req: Request): Option[Row] =
  var key = ""
  if req.headers.hasKey("x-api-key"): key = req.headers["x-api-key"]
  elif req.headers.hasKey("authorization"):
    let a = req.headers["authorization"]
    if a.startsWith("Bearer "): key = a[7 .. ^1]
    else: key = a
  if key.len == 0:
    for pair in decodeQuery(req.url.query):
      if pair[0] == "api_key": key = pair[1]
  if key.len == 0: key = getEnv("AGENT_DEFAULT_KEY", "modular-agent-secret-key")
  let rows = store.query("SELECT * FROM tenants WHERE api_key_hash=?", @[%sha1Hex(key)])
  if rows.len > 0: return some(rows[0])
  return none(Row)

proc ensureDefaultTenant(): Row =
  let key = getEnv("AGENT_DEFAULT_KEY", "modular-agent-secret-key")
  let kh = sha1Hex(key)
  var tArr = newJArray()
  for k, _ in toolRegistry: tArr.add(%k)
  let rows = store.query("SELECT * FROM tenants WHERE name='default'", @[])
  if rows.len > 0:
    let existingTools = getJson(rows[0], "allowed_tools")
    if existingTools.kind != JArray or existingTools.len == 0:
      discard store.exec("UPDATE tenants SET allowed_tools=?, api_key_hash=? WHERE tenant_id=?",
        @[%($tArr), %kh, %getStr(rows[0], "tenant_id")])
      let refreshed = store.query("SELECT * FROM tenants WHERE tenant_id=?", @[%getStr(rows[0], "tenant_id")])
      return refreshed[0]
    if getStr(rows[0], "api_key_hash") != kh:
      discard store.exec("UPDATE tenants SET api_key_hash=? WHERE tenant_id=?", @[%kh, %getStr(rows[0], "tenant_id")])
      let refreshed = store.query("SELECT * FROM tenants WHERE tenant_id=?", @[%getStr(rows[0], "tenant_id")])
      return refreshed[0]
    return rows[0]
  let tid = newId("tenant")
  discard store.exec(
    "INSERT INTO tenants (tenant_id, name, api_key_hash, token_budget, tokens_used, allowed_tools, created_at) " &
    "VALUES (?,?,?,?,0,?,?)",
    @[%tid, %"default", %kh, %TokenBudgetDefault, %($tArr), %nowF()])
  let fresh = store.query("SELECT * FROM tenants WHERE tenant_id=?", @[%tid])
  return fresh[0]

proc seedDefaultSkills(tenantId: string) =
  let rows = store.query("SELECT COUNT(*) AS n FROM skills WHERE tenant_id=?", @[%tenantId])
  if rows.len > 0 and getInt(rows[0], "n") > 0: return
  proc addSkill(name, domain, trigger, procedure: string) =
    let sid = newId("skill")
    let emb = embToJson(textEmbedding(name & " " & domain & " " & trigger & " " & procedure))
    discard store.exec(
      "INSERT INTO skills (skill_id, tenant_id, name, domain, trigger_spec, procedure_spec, " &
      "preconditions_json, postconditions_json, failure_modes_json, version, active, success_count, " &
      "failure_count, reward, embedding_json, created_at, updated_at) " &
      "VALUES (?,?,?,?,?,?,'[]','[]','[]',1,1,0,0,0.0,?,?,?)",
      @[%sid, %tenantId, %name, %domain, %trigger, %procedure, %emb, %nowF(), %nowF()])
  addSkill("filesystem_operations", "filesystem",
           "task modifies or verifies files in workspace",
           "1) list_dir to inspect workspace. 2) read_file or check_lines to establish ground truth. 3) write_file or append_file. 4) check_lines to confirm changes. 5) record paths into artifacts.")
  addSkill("arithmetic_verification", "computation",
           "numerical calculations or formula evaluation",
           "Use math_eval to evaluate mathematical expressions deterministically without hallucination. Save answers to facts.")
  addSkill("knowledge_retrieval", "memory",
           "facts or patterns not present in working memory",
           "1) memory_search wiki and skills. 2) Record atomic discoveries to facts. 3) If reusable workaround found, call memory_write to update playbook.")


proc consolidateKnowledgeOnce() =
  let tenants = store.query("SELECT tenant_id FROM tenants", @[])
  for tr in tenants:
    let tenantId = getStr(tr, "tenant_id")
    let refs = store.query("SELECT failure_point, pivot_action, attribution, patch_json, created_at FROM reflections WHERE tenant_id=? ORDER BY created_at DESC LIMIT 20", @[%tenantId])
    if refs.len == 0: continue
    var bodyParts: seq[string] = @[]
    for r in refs:
      bodyParts.add("Failure: " & getStr(r, "failure_point") & "\nAttribution: " & getStr(r, "attribution") & "\nPivot: " & getStr(r, "pivot_action") & "\nPatch: " & getStr(r, "patch_json"))
    try:
      discard commitKnowledgeDoc(tenantId, "verified-failure-patterns", "reflections", bodyParts.join("\n\n---\n\n"))
    except CatchableError:
      discard

proc knowledgeConsolidationLoop() {.async.} =
  while true:
    await sleepAsync(300000)
    consolidateKnowledgeOnce()

proc handleSse(req: Request, tenantId: string) {.async.} =
  let client = SseClient(req: req, tenantId: tenantId, alive: true, queue: initDeque[string]())
  initLock(client.lock)
  acquire(sseLock)
  sseClients.add(client)
  release(sseLock)
  var headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
  try:
    await req.client.send(headers)
    await req.client.send("data: " & canonical(%*{"type": "connected", "tenant_id": tenantId, "time": nowF()}) & "\n\n")
    var lastPing = nowF()
    while client.alive and not req.client.isClosed():
      var batch: seq[string] = @[]
      acquire(client.lock)
      while client.queue.len > 0: batch.add(client.queue.popFirst())
      release(client.lock)
      for payload in batch: await req.client.send(payload)
      if nowF() - lastPing > 15.0:
        lastPing = nowF()
        await req.client.send(": ping\n\n")
      await sleepAsync(100)
  except CatchableError: discard
  finally:
    client.alive = false
    acquire(sseLock)
    var live: seq[SseClient] = @[]
    for c in sseClients:
      if c != client: live.add(c)
    sseClients = live
    release(sseLock)

proc respondJson(req: Request, code: HttpCode, body: JsonNode) {.async.} =
  let headers = newHttpHeaders({
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Api-Key"
  })
  await req.respond(code, canonical(body), headers)

proc handleHttpRequest(req: Request) {.async, gcsafe.} =
  if req.reqMethod == HttpOptions:
    let headers = newHttpHeaders({
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Api-Key"
    })
    await req.respond(Http204, "", headers)
    return

  let path = req.url.path
  if path in ["/", "/index.html"]:
    for candidate in ["index.html", "public/index.html", "static/index.html"]:
      if fileExists(candidate):
        let content = readFile(candidate)
        let headers = newHttpHeaders({"Content-Type": "text/html; charset=utf-8", "Access-Control-Allow-Origin": "*"})
        await req.respond(Http200, content, headers)
        return
    let fallbackHtml = "<!DOCTYPE html><html><head><title>Autonomous Agent Runtime</title></head><body style=\"font-family:sans-serif;padding:30px;background:#0d1117;color:#c9d1d9;\"><h2>Autonomous Agent Runtime</h2><p>Modular GLM-5.3 Runtime Operational.</p></body></html>"
    let headers = newHttpHeaders({"Content-Type": "text/html; charset=utf-8", "Access-Control-Allow-Origin": "*"})
    await req.respond(Http200, fallbackHtml, headers)
    return

  if path == "/api/health":
    await respondJson(req, Http200, %*{"status": "healthy", "model": ModularModel, "fts5": fts5Available, "time": nowF()})
    return

  let tOpt = authenticate(req)
  if tOpt.isNone:
    await respondJson(req, Http401, %*{"error": "unauthorized"})
    return
  let tenant = tOpt.get()
  let tenantId = getStr(tenant, "tenant_id")

  if path == "/api/events" and req.reqMethod == HttpGet:
    await handleSse(req, tenantId)
    return

  if path == "/api/runs" and req.reqMethod == HttpPost:
    var body = newJObject()
    try:
      body = parseJson(req.body)
    except CatchableError:
      await respondJson(req, Http400, %*{"error": "invalid JSON body"})
      return
    let title = body{"title"}.getStr(body{"goal"}.getStr(body{"message"}.getStr("autonomous task")))
    let spec = if body.hasKey("spec") and body["spec"].kind == JObject: body["spec"] else: %*{"goal": title}
    let maxSteps = body{"max_steps"}.getInt(MaxStepsDefault).int
    try:
      let h = createTask(tenant, title, spec, maxSteps)
      attachBroadcast(h)
      discard launchTask(h)
      await respondJson(req, Http201, %*{"task_id": h.taskId, "status": h.status, "goal": title})
    except ValueError as e:
      await respondJson(req, Http400, %*{"error": e.msg})
    except CatchableError as e:
      await respondJson(req, Http500, %*{"error": "task creation failed", "detail": e.msg})
    return

  if path == "/api/runs" and req.reqMethod == HttpGet:
    let rows = store.query("SELECT * FROM tasks WHERE tenant_id=? ORDER BY updated_at DESC LIMIT 100", @[%tenantId])
    var arr = newJArray()
    for r in rows:
      arr.add(%*{
        "task_id": getStr(r, "task_id"), "title": getStr(r, "title"), "status": getStr(r, "status"),
        "step_index": getInt(r, "step_index"), "max_steps": getInt(r, "max_steps"),
        "verified": getInt(r, "verified") == 1, "tokens_used": getInt(r, "tokens_used"),
        "created_at": getFloat(r, "created_at"), "updated_at": getFloat(r, "updated_at")
      })
    await respondJson(req, Http200, %*{"runs": arr})
    return

  if path.startsWith("/api/runs/"):
    let rest = path[10 .. ^1]
    let parts = rest.split('/')
    let taskId = parts[0]
    let h = restoreTask(taskId)
    if h == nil or h.tenantId != tenantId:
      await respondJson(req, Http404, %*{"error": "task not found"})
      return

    if parts.len == 1 and req.reqMethod == HttpGet:
      acquire(h.lock)
      let payload = %*{
        "task_id": h.taskId, "title": h.title, "status": h.status, "step_index": h.stepIndex,
        "max_steps": h.maxSteps, "verified": h.verified, "terminal_reason": h.terminalReason,
        "paused": h.paused, "transition_busy": h.transitionBusy,
        "sigma": copy(h.sigma), "obs": copy(h.obs), "cognition": copy(h.cognition)
      }
      release(h.lock)
      let (_, verificationReport) = verifyTerminal(h)
      payload["verification_report"] = verificationReport
      payload["loop_active"] = %h.loopActive.load(moAcquire)
      await respondJson(req, Http200, payload)
      return

    if parts.len == 2 and parts[1] == "stop" and req.reqMethod == HttpPost:
      acquire(h.lock)
      h.stopRequested = true
      h.paused = false
      h.status = "halted"
      h.terminalReason = "stopped by operator"
      release(h.lock)
      h.checkpoint(newJObject(), newJObject(), %*{"operator": "stop"})
      await respondJson(req, Http200, %*{"task_id": h.taskId, "status": "halted"})
      return

    if parts.len == 2 and parts[1] == "pause" and req.reqMethod == HttpPost:
      acquire(h.lock)
      h.paused = true
      let pauseStatus = h.status
      release(h.lock)
      await respondJson(req, Http200, %*{"task_id": h.taskId, "status": pauseStatus, "paused": true})
      return

    if parts.len == 2 and parts[1] == "resume" and req.reqMethod == HttpPost:
      var resumable = true
      acquire(h.lock)
      if h.verified or h.status == "succeeded" or h.stepIndex >= h.maxSteps:
        resumable = false
      else:
        h.paused = false
        h.stopRequested = false
        if h.status in ["halted", "failed", "queued"]:
          h.status = "running"
          h.terminalReason = ""
      let resumeStatus = h.status
      release(h.lock)
      if not resumable:
        await respondJson(req, Http409, %*{"error": "task is terminal or max_steps reached", "task_id": h.taskId, "status": resumeStatus})
        return
      discard launchTask(h)
      await respondJson(req, Http200, %*{"task_id": h.taskId, "status": h.status, "paused": false, "loop_active": h.loopActive.load(moAcquire)})
      return

    if parts.len == 2 and parts[1] == "message" and req.reqMethod == HttpPost:
      var body = newJObject()
      try:
        body = parseJson(req.body)
      except CatchableError:
        await respondJson(req, Http400, %*{"error": "invalid JSON body"})
        return
      let msg = boundUtf8Bytes(body{"message"}.getStr(body{"content"}.getStr("")), 32768)
      if msg.len == 0:
        await respondJson(req, Http400, %*{"error": "message required"})
        return
      var canLaunch = false
      acquire(h.lock)
      if h.verified or h.status == "succeeded" or h.stepIndex >= h.maxSteps:
        release(h.lock)
        await respondJson(req, Http409, %*{"error": "task is terminal or max_steps reached", "task_id": h.taskId})
        return
      var obs = copy(h.obs)
      obs["operator_message"] = %msg
      obs["operator_message_at"] = %nowF()
      h.obs = obs
      h.paused = false
      if h.status in ["halted", "failed", "queued"]:
        h.status = "running"
        h.stopRequested = false
        h.terminalReason = ""
        canLaunch = true
      else:
        canLaunch = not h.loopActive.load(moAcquire)
      release(h.lock)
      h.checkpoint(newJObject(), newJObject(), %*{"operator_message": msg})
      if canLaunch: discard launchTask(h)
      await respondJson(req, Http200, %*{"task_id": h.taskId, "injected": true, "status": h.status})
      return

    if parts.len == 2 and parts[1] == "traces" and req.reqMethod == HttpGet:
      let rows = store.query("SELECT * FROM raw_traces WHERE task_id=? ORDER BY step_index ASC LIMIT 200", @[%taskId])
      var arr = newJArray()
      for r in rows:
        arr.add(%*{
          "trace_id": getStr(r, "trace_id"), "step_index": getInt(r, "step_index"),
          "action": getJson(r, "action_json"), "obs": getJson(r, "obs_json"),
          "success": getInt(r, "success") == 1, "latency_ms": getInt(r, "latency_ms"),
          "immutable_hash": getStr(r, "immutable_hash")
        })
      await respondJson(req, Http200, %*{"traces": arr})
      return

  if path == "/api/skills" and req.reqMethod == HttpGet:
    let rows = store.query("SELECT * FROM skills WHERE tenant_id=? AND active=1 ORDER BY reward DESC", @[%tenantId])
    var arr = newJArray()
    for r in rows:
      arr.add(%*{
        "skill_id": getStr(r, "skill_id"), "name": getStr(r, "name"), "domain": getStr(r, "domain"),
        "trigger": getStr(r, "trigger_spec"), "procedure": getStr(r, "procedure_spec"),
        "success_count": getInt(r, "success_count"), "reward": getFloat(r, "reward")
      })
    await respondJson(req, Http200, %*{"skills": arr})
    return

  if path == "/api/skills" and req.reqMethod == HttpPost:
    var body = newJObject()
    try:
      body = parseJson(req.body)
    except CatchableError:
      await respondJson(req, Http400, %*{"error": "invalid JSON body"})
      return
    let name = body{"name"}.getStr("")
    let domain = body{"domain"}.getStr("general")
    let trigger = body{"trigger"}.getStr("")
    let procedure = body{"procedure"}.getStr("")
    if name.len == 0 or procedure.len == 0:
      await respondJson(req, Http400, %*{"error": "name and procedure required"})
      return
    acquire(skillGateLock)
    if skillGateBusy:
      release(skillGateLock)
      await respondJson(req, Http409, %*{"error": "skill regression gate already running"})
      return
    skillGateBusy = true
    release(skillGateLock)
    var oldRows: seq[Row] = @[]
    var mutationApplied = false
    try:
      let beforeReport = await runRegressionGate(tenantId)
      oldRows = store.query("SELECT * FROM skills WHERE tenant_id=? AND name=?", @[%tenantId, %name])
      let emb = embToJson(textEmbedding(name & " " & domain & " " & trigger & " " & procedure))
      let preconditions = if body.hasKey("preconditions") and body["preconditions"].kind == JArray: canonical(body["preconditions"]) else: "[]"
      let postconditions = if body.hasKey("postconditions") and body["postconditions"].kind == JArray: canonical(body["postconditions"]) else: "[]"
      let failureModes = if body.hasKey("failure_modes") and body["failure_modes"].kind == JArray: canonical(body["failure_modes"]) else: "[]"
      let ts = nowF()
      var sid = ""
      if oldRows.len > 0:
        sid = getStr(oldRows[0], "skill_id")
        discard store.exec(
          "UPDATE skills SET domain=?, trigger_spec=?, procedure_spec=?, preconditions_json=?, postconditions_json=?, " &
          "failure_modes_json=?, version=version+1, active=1, embedding_json=?, updated_at=? WHERE skill_id=? AND tenant_id=?",
          @[%domain, %trigger, %procedure, %preconditions, %postconditions, %failureModes, %emb, %ts, %sid, %tenantId])
      else:
        sid = newId("skill")
        discard store.exec(
          "INSERT INTO skills (skill_id, tenant_id, name, domain, trigger_spec, procedure_spec, preconditions_json, " &
          "postconditions_json, failure_modes_json, version, active, success_count, failure_count, reward, embedding_json, created_at, updated_at) " &
          "VALUES (?,?,?,?,?,?,?,?,?,1,1,0,0,0.0,?,?,?)",
          @[%sid, %tenantId, %name, %domain, %trigger, %procedure, %preconditions, %postconditions, %failureModes, %emb, %ts, %ts])
      mutationApplied = true
      let afterReport = await runRegressionGate(tenantId)
      let beforeScore = beforeReport{"score"}.getFloat(0.0)
      let afterScore = afterReport{"score"}.getFloat(0.0)
      if afterScore + 1e-12 < beforeScore:
        if oldRows.len > 0:
          let old = oldRows[0]
          discard store.exec(
            "UPDATE skills SET name=?, domain=?, trigger_spec=?, procedure_spec=?, preconditions_json=?, postconditions_json=?, " &
            "failure_modes_json=?, version=?, active=?, success_count=?, failure_count=?, reward=?, embedding_json=?, created_at=?, updated_at=? " &
            "WHERE skill_id=? AND tenant_id=?",
            @[%getStr(old, "name"), %getStr(old, "domain"), %getStr(old, "trigger_spec"), %getStr(old, "procedure_spec"),
              %getStr(old, "preconditions_json", "[]"), %getStr(old, "postconditions_json", "[]"), %getStr(old, "failure_modes_json", "[]"),
              %getInt(old, "version", 1), %getInt(old, "active", 1), %getInt(old, "success_count", 0), %getInt(old, "failure_count", 0),
              %getFloat(old, "reward", 0.0), %getStr(old, "embedding_json", "[]"), %getFloat(old, "created_at", ts), %getFloat(old, "updated_at", ts),
              %getStr(old, "skill_id"), %tenantId])
        else:
          discard store.exec("DELETE FROM skills WHERE skill_id=? AND tenant_id=?", @[%sid, %tenantId])
        mutationApplied = false
        await respondJson(req, Http409, %*{"error": "regression gate rejected skill patch", "before": beforeReport, "after": afterReport})
        return
      let activeRows = store.query("SELECT skill_id, version FROM skills WHERE tenant_id=? AND name=?", @[%tenantId, %name])
      let finalSid = if activeRows.len > 0: getStr(activeRows[0], "skill_id", sid) else: sid
      let finalVersion = if activeRows.len > 0: getInt(activeRows[0], "version", 1) else: 1
      await respondJson(req, Http201, %*{"skill_id": finalSid, "version": finalVersion, "before": beforeReport, "after": afterReport})
      return
    except CatchableError as e:
      if mutationApplied:
        try:
          let current = store.query("SELECT skill_id FROM skills WHERE tenant_id=? AND name=?", @[%tenantId, %name])
          if oldRows.len > 0:
            let old = oldRows[0]
            discard store.exec(
              "UPDATE skills SET name=?, domain=?, trigger_spec=?, procedure_spec=?, preconditions_json=?, postconditions_json=?, " &
              "failure_modes_json=?, version=?, active=?, success_count=?, failure_count=?, reward=?, embedding_json=?, created_at=?, updated_at=? " &
              "WHERE skill_id=? AND tenant_id=?",
              @[%getStr(old, "name"), %getStr(old, "domain"), %getStr(old, "trigger_spec"), %getStr(old, "procedure_spec"),
                %getStr(old, "preconditions_json", "[]"), %getStr(old, "postconditions_json", "[]"), %getStr(old, "failure_modes_json", "[]"),
                %getInt(old, "version", 1), %getInt(old, "active", 1), %getInt(old, "success_count", 0), %getInt(old, "failure_count", 0),
                %getFloat(old, "reward", 0.0), %getStr(old, "embedding_json", "[]"), %getFloat(old, "created_at", nowF()), %getFloat(old, "updated_at", nowF()),
                %getStr(old, "skill_id"), %tenantId])
          elif current.len > 0:
            discard store.exec("DELETE FROM skills WHERE skill_id=? AND tenant_id=?", @[%getStr(current[0], "skill_id"), %tenantId])
        except CatchableError:
          discard
      await respondJson(req, Http500, %*{"error": "skill regression gate failed", "detail": e.msg})
      return
    finally:
      acquire(skillGateLock)
      skillGateBusy = false
      release(skillGateLock)

  if path == "/api/diagnostics/run" and req.reqMethod == HttpGet:
    let rep = await runRegressionGate(tenantId)
    await respondJson(req, Http200, rep)
    return

  if path == "/api/tools" and req.reqMethod == HttpGet:
    var allowedSet = initHashSet[string]()
    let aj = getJson(tenant, "allowed_tools")
    if aj.kind == JArray:
      for it in aj.elems: allowedSet.incl(it.getStr())
    await respondJson(req, Http200, %*{"tools": toolCatalog(allowedSet)})
    return

  await respondJson(req, Http404, %*{"error": "endpoint not found"})

proc main() =
  randomize()
  initLock(rngLock)
  initLock(tasksLock)
  initLock(sseLock)
  initLock(skillGateLock)
  globalRng = initRand(int64(epochTime() * 1_000_000.0))
  createDir(WorkspaceRoot)
  createDir(KnowledgeRoot)
  store = openStore(DbFile)
  migrate(store)
  initTools()
  let defaultTenant = ensureDefaultTenant()
  seedDefaultSkills(getStr(defaultTenant, "tenant_id"))
  resumePendingTasks()
  asyncCheck knowledgeConsolidationLoop()
  let server = newAsyncHttpServer(maxBody = 33_554_432)
  echo "Runtime listening on port ", ServerPort
  waitFor server.serve(Port(ServerPort), handleHttpRequest, address = "0.0.0.0")

when isMainModule:
  main()