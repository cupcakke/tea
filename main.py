import os
import sys
import json
import time
import math
import uuid
import hmac
import hashlib
import base64
import sqlite3
import asyncio
import logging
import threading
import subprocess
import shutil
import tempfile
import re
import struct
import traceback
import contextlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable, Set, Union
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, Header, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import uvicorn

from openai import OpenAI


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format=LOG_FORMAT, stream=sys.stdout)
log = logging.getLogger("agent.runtime")


BASE_DIR = Path(os.environ.get("AGENT_HOME", Path.cwd() / ".agent_runtime")).resolve()
DB_PATH = BASE_DIR / "runtime.db"
WORKSPACE_ROOT = BASE_DIR / "workspaces"
WIKI_ROOT = BASE_DIR / "wiki"
SKILL_ROOT = BASE_DIR / "skills"
TRACE_ROOT = BASE_DIR / "traces"
DISTILL_ROOT = BASE_DIR / "distill"
STATIC_ROOT = Path(os.environ.get("AGENT_STATIC", Path.cwd())).resolve()

for _p in (BASE_DIR, WORKSPACE_ROOT, WIKI_ROOT, SKILL_ROOT, TRACE_ROOT, DISTILL_ROOT):
    _p.mkdir(parents=True, exist_ok=True)


MODEL_NAME = os.environ.get("AGENT_MODEL", "zai-org/glm-5.3")
MODEL_BASE_URL = os.environ.get("MODULAR_BASE_URL", "https://api.modular.com/v1")
MODEL_API_KEY = os.environ.get("MODULAR_API_KEY", "")
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "100000"))
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.96"))
TOP_P = float(os.environ.get("AGENT_TOP_P", "1"))
FREQUENCY_PENALTY = float(os.environ.get("AGENT_FREQ_PENALTY", "0.8"))
PRESENCE_PENALTY = float(os.environ.get("AGENT_PRES_PENALTY", "0.5"))
SEED = int(os.environ.get("AGENT_SEED", "1234"))

SYSTEM1_HZ = float(os.environ.get("AGENT_SYSTEM1_HZ", "20"))
SYSTEM2_HZ = float(os.environ.get("AGENT_SYSTEM2_HZ", "1"))
COGNITION_K = int(os.environ.get("AGENT_COGNITION_K", "8"))
COGNITION_H = int(os.environ.get("AGENT_COGNITION_H", "64"))
EMBED_DIM = int(os.environ.get("AGENT_EMBED_DIM", "512"))

DEFAULT_STEP_BUDGET = int(os.environ.get("AGENT_STEP_BUDGET", "512"))
DEFAULT_TOKEN_BUDGET = int(os.environ.get("AGENT_TOKEN_BUDGET", "4000000"))
DEFAULT_WALL_BUDGET_S = int(os.environ.get("AGENT_WALL_BUDGET", "86400"))

SIGNING_SECRET = os.environ.get("AGENT_SIGNING_SECRET", "modular-agent-runtime-signing-key")
ADMIN_TOKEN = os.environ.get("AGENT_ADMIN_TOKEN", "admin-local-token")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime] = None) -> str:
    return (dt or utcnow()).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sign_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hmac.new(SIGNING_SECRET.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def verify_signature(payload: Any, signature: str) -> bool:
    return hmac.compare_digest(sign_payload(payload), signature or "")


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def jload(text: Optional[str], default: Any = None) -> Any:
    if text is None or text == "":
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 3.6))


class SQLiteStore:
    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=60.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._connect()
            self._local.conn = c
        return c

    @contextlib.contextmanager
    def tx(self):
        with self._write_lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
                c.execute("COMMIT")
            except Exception:
                try:
                    c.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def query(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        cur = self.conn.execute(sql, tuple(params))
        rows = cur.fetchall()
        cur.close()
        return rows

    def one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self.tx() as c:
            c.execute(sql, tuple(params))

    def executemany(self, sql: str, seq: Iterable[Iterable[Any]]) -> None:
        with self.tx() as c:
            c.executemany(sql, [tuple(x) for x in seq])

    def _init_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS tenants (
            tenant_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            api_key_hash TEXT NOT NULL,
            token_budget INTEGER NOT NULL DEFAULT 4000000,
            tokens_used INTEGER NOT NULL DEFAULT 0,
            allowed_tools TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_conv_tenant ON conversations(tenant_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            meta TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            seq INTEGER NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, seq ASC);

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            conversation_id TEXT,
            spec TEXT NOT NULL,
            status TEXT NOT NULL,
            step INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT,
            terminal_state TEXT,
            verdict TEXT,
            tokens_used INTEGER NOT NULL DEFAULT 0,
            wall_ms INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            lease_owner TEXT,
            lease_expires_at TEXT,
            resume_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_runs_tenant ON runs(tenant_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            node TEXT NOT NULL,
            sigma TEXT NOT NULL,
            observation TEXT NOT NULL DEFAULT '{}',
            pending TEXT NOT NULL DEFAULT '{}',
            digest TEXT NOT NULL,
            signature TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ckpt_run_step ON checkpoints(run_id, step, node);
        CREATE INDEX IF NOT EXISTS idx_ckpt_run ON checkpoints(run_id, step DESC);

        CREATE TABLE IF NOT EXISTS raw_traces (
            trace_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            kind TEXT NOT NULL,
            skill_id TEXT,
            pre_state TEXT NOT NULL DEFAULT '{}',
            action TEXT NOT NULL DEFAULT '{}',
            outcome TEXT NOT NULL DEFAULT '{}',
            state_delta TEXT NOT NULL DEFAULT '{}',
            receipt TEXT NOT NULL DEFAULT '{}',
            success INTEGER NOT NULL DEFAULT 0,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            digest TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_trace_run ON raw_traces(run_id, step ASC);
        CREATE INDEX IF NOT EXISTS idx_trace_skill ON raw_traces(skill_id, success);

        CREATE TABLE IF NOT EXISTS working_memory (
            wm_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            progress TEXT NOT NULL DEFAULT '[]',
            open_goals TEXT NOT NULL DEFAULT '[]',
            dependencies TEXT NOT NULL DEFAULT '[]',
            constraints TEXT NOT NULL DEFAULT '[]',
            facts TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_wm_run ON working_memory(run_id);

        CREATE TABLE IF NOT EXISTS skills (
            skill_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            name TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            category TEXT NOT NULL DEFAULT 'general',
            summary TEXT NOT NULL,
            preconditions TEXT NOT NULL DEFAULT '[]',
            procedure TEXT NOT NULL DEFAULT '[]',
            failure_modes TEXT NOT NULL DEFAULT '[]',
            tags TEXT NOT NULL DEFAULT '[]',
            embedding BLOB,
            uses INTEGER NOT NULL DEFAULT 0,
            successes INTEGER NOT NULL DEFAULT 0,
            failures INTEGER NOT NULL DEFAULT 0,
            score REAL NOT NULL DEFAULT 0.5,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_skill_tenant ON skills(tenant_id, status);

        CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
            skill_id UNINDEXED,
            tenant_id UNINDEXED,
            name,
            summary,
            procedure,
            tags,
            tokenize='porter unicode61'
        );

        CREATE TABLE IF NOT EXISTS skill_patches (
            patch_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            target_skill_id TEXT,
            component TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            proposal TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'proposed',
            gate_report TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            decided_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_patch_status ON skill_patches(status, created_at DESC);

        CREATE TABLE IF NOT EXISTS wiki_pages (
            page_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            slug TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            body TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            embedding BLOB,
            updated_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_wiki_slug ON wiki_pages(tenant_id, slug);

        CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
            page_id UNINDEXED,
            tenant_id UNINDEXED,
            title,
            body,
            category,
            tokenize='porter unicode61'
        );

        CREATE TABLE IF NOT EXISTS wiki_revisions (
            revision_id TEXT PRIMARY KEY,
            page_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            diff TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_wikirev_page ON wiki_revisions(page_id, version DESC);

        CREATE TABLE IF NOT EXISTS reflection_patches (
            reflection_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            verdict TEXT NOT NULL,
            failure_points TEXT NOT NULL DEFAULT '[]',
            pivot_actions TEXT NOT NULL DEFAULT '[]',
            patch_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_refl_run ON reflection_patches(run_id);

        CREATE TABLE IF NOT EXISTS distill_samples (
            sample_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            student_prompt TEXT NOT NULL,
            teacher_prompt TEXT NOT NULL,
            action_text TEXT NOT NULL,
            student_logprobs TEXT NOT NULL DEFAULT '[]',
            teacher_logprobs TEXT NOT NULL DEFAULT '[]',
            tokens TEXT NOT NULL DEFAULT '[]',
            reverse_kl REAL NOT NULL DEFAULT 0.0,
            advantage REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_distill_run ON distill_samples(run_id, step ASC);

        CREATE TABLE IF NOT EXISTS policy_weights (
            weight_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            feature TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 0.0,
            updates INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pw ON policy_weights(tenant_id, feature);

        CREATE TABLE IF NOT EXISTS cognition_tokens (
            cog_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            vector BLOB NOT NULL,
            gate REAL NOT NULL DEFAULT 1.0,
            subgoal TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            created_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cog_run ON cognition_tokens(run_id, step DESC);

        CREATE TABLE IF NOT EXISTS diagnostic_tasks (
            task_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            name TEXT NOT NULL,
            spec TEXT NOT NULL,
            verifier TEXT NOT NULL,
            baseline_score REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            audit_id TEXT PRIMARY KEY,
            tenant_id TEXT,
            run_id TEXT,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT '{}',
            allowed INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            run_id TEXT,
            tenant_id TEXT,
            conversation_id TEXT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            seq INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq ASC);
        CREATE INDEX IF NOT EXISTS idx_events_conv ON events(conversation_id, seq ASC);

        CREATE TABLE IF NOT EXISTS counters (
            name TEXT PRIMARY KEY,
            value INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS file_index (
            file_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            run_id TEXT,
            rel_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_file_path ON file_index(tenant_id, rel_path);
        """
        with self._write_lock:
            c = self.conn
            c.executescript(ddl)

    def next_seq(self, name: str) -> int:
        with self.tx() as c:
            c.execute("INSERT INTO counters(name, value) VALUES(?, 0) ON CONFLICT(name) DO NOTHING", (name,))
            c.execute("UPDATE counters SET value = value + 1 WHERE name = ?", (name,))
            row = c.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()
            return int(row[0])

    def list_policy_hints(self, tenant_id: str, limit: int = 24) -> List[Dict[str, Any]]:
        rows = self.query(
            "SELECT feature, weight, updates, updated_at FROM policy_weights WHERE tenant_id = ? AND weight > 0 ORDER BY weight DESC, updates DESC LIMIT ?",
            (tenant_id, int(limit)),
        )
        return [dict(row) for row in rows]


STORE = SQLiteStore(DB_PATH)


def pack_vector(vec: List[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: Optional[bytes]) -> List[float]:
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob[: n * 4]))


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


class HashingEmbedder:
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def _hash(self, token: str, salt: int) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=struct.pack("<I", salt)).digest()
        return struct.unpack("<Q", h)[0]

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        if not toks:
            return vec
        counts: Dict[str, int] = defaultdict(int)
        for t in toks:
            counts[t] += 1
        for i in range(len(toks) - 1):
            counts[toks[i] + "_" + toks[i + 1]] += 1
        for tok, cnt in counts.items():
            for salt in (11, 29):
                hv = self._hash(tok, salt)
                idx = hv % self.dim
                sign = 1.0 if ((hv >> 33) & 1) == 0 else -1.0
                vec[idx] += sign * (1.0 + math.log(cnt))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(n):
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        if na <= 0 or nb <= 0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))


EMBEDDER = HashingEmbedder(EMBED_DIM)


def fts_escape(query: str) -> str:
    toks = tokenize(query)
    toks = [t for t in toks if len(t) > 1][:24]
    if not toks:
        return ""
    return " OR ".join(f'"{t}"' for t in toks)


def reciprocal_rank_fusion(rankings: List[List[str]], k: float = 60.0, weights: Optional[List[float]] = None) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    for i, ranking in enumerate(rankings):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, key in enumerate(ranking):
            scores[key] += w * (1.0 / (k + rank + 1.0))
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class SecurityError(Exception):
    pass


class ValidationError(Exception):
    pass


class BudgetExceeded(Exception):
    pass


class ToolDenied(Exception):
    pass


@dataclass
class Tenant:
    tenant_id: str
    name: str
    token_budget: int
    tokens_used: int
    allowed_tools: List[str]
    active: bool

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Tenant":
        return Tenant(
            tenant_id=row["tenant_id"],
            name=row["name"],
            token_budget=int(row["token_budget"]),
            tokens_used=int(row["tokens_used"]),
            allowed_tools=jload(row["allowed_tools"], []) or [],
            active=bool(row["active"]),
        )


class TenantRegistry:
    DEFAULT_TOOLS = [
        "workspace.write_file",
        "workspace.read_file",
        "workspace.append_file",
        "workspace.replace_lines",
        "workspace.check_lines",
        "workspace.list_dir",
        "workspace.delete_file",
        "memory.wiki_search",
        "memory.wiki_write",
        "memory.skill_search",
        "memory.skill_upsert",
        "memory.trace_query",
        "compute.python",
        "compute.shell",
        "compute.http_get",
        "reason.think",
        "control.finish",
        "control.fail",
    ]

    def __init__(self, store: SQLiteStore):
        self.store = store
        self._cache: Dict[str, Tenant] = {}
        self._lock = threading.RLock()
        self.ensure_tenant("public", "Public", os.environ.get("AGENT_PUBLIC_KEY", "public-key"))

    @staticmethod
    def hash_key(key: str) -> str:
        return hashlib.sha256(("agentsalt::" + key).encode("utf-8")).hexdigest()

    def ensure_tenant(self, tenant_id: str, name: str, api_key: str, token_budget: int = DEFAULT_TOKEN_BUDGET) -> Tenant:
        with self._lock:
            row = self.store.one("SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,))
            if row is None:
                self.store.execute(
                    "INSERT INTO tenants(tenant_id, name, api_key_hash, token_budget, tokens_used, allowed_tools, created_at, active) "
                    "VALUES(?,?,?,?,?,?,?,1)",
                    (tenant_id, name, self.hash_key(api_key), token_budget, 0, jdump(self.DEFAULT_TOOLS), iso()),
                )
                row = self.store.one("SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,))
            t = Tenant.from_row(row)
            self._cache[tenant_id] = t
            return t

    def get(self, tenant_id: str) -> Tenant:
        row = self.store.one("SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,))
        if row is None:
            raise SecurityError(f"unknown tenant {tenant_id}")
        t = Tenant.from_row(row)
        if not t.active:
            raise SecurityError(f"tenant {tenant_id} disabled")
        return t

    def authenticate(self, tenant_id: Optional[str], api_key: Optional[str]) -> Tenant:
        tid = tenant_id or "public"
        row = self.store.one("SELECT * FROM tenants WHERE tenant_id = ?", (tid,))
        if row is None:
            raise SecurityError("invalid tenant")
        t = Tenant.from_row(row)
        if not t.active:
            raise SecurityError("tenant disabled")
        if tid != "public":
            if not api_key or not hmac.compare_digest(self.hash_key(api_key), row["api_key_hash"]):
                raise SecurityError("invalid credentials")
        return t

    def charge_tokens(self, tenant_id: str, tokens: int) -> None:
        with self.store.tx() as c:
            row = c.execute("SELECT token_budget, tokens_used FROM tenants WHERE tenant_id = ?", (tenant_id,)).fetchone()
            if row is None:
                raise SecurityError("unknown tenant")
            budget = int(row[0])
            used = int(row[1]) + max(0, tokens)
            c.execute("UPDATE tenants SET tokens_used = ? WHERE tenant_id = ?", (used, tenant_id))
            if used > budget:
                raise BudgetExceeded(f"tenant {tenant_id} token budget exhausted ({used}/{budget})")

    def authorize_tool(self, tenant_id: str, tool: str) -> None:
        t = self.get(tenant_id)
        if tool not in t.allowed_tools:
            raise ToolDenied(f"tool {tool} not authorized for tenant {tenant_id}")


TENANTS = TenantRegistry(STORE)


def audit(tenant_id: Optional[str], run_id: Optional[str], actor: str, action: str, detail: Any, allowed: bool = True) -> None:
    try:
        STORE.execute(
            "INSERT INTO audit_log(audit_id, tenant_id, run_id, actor, action, detail, allowed, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (new_id("aud"), tenant_id, run_id, actor, action, jdump(detail), 1 if allowed else 0, iso()),
        )
    except Exception as exc:
        log.warning("audit failure: %s", exc)


class EventBus:
    def __init__(self):
        self._subs: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, topic: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4096)
        with self._lock:
            self._subs[topic].add(q)
        return q

    def unsubscribe(self, topic: str, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs[topic].discard(q)
            if not self._subs[topic]:
                self._subs.pop(topic, None)

    def _deliver(self, topic: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            queues = list(self._subs.get(topic, ()))
        for q in queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    pass

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            self._deliver(topic, payload)
            return
        try:
            loop.call_soon_threadsafe(self._deliver, topic, payload)
        except RuntimeError:
            self._deliver(topic, payload)


BUS = EventBus()


def emit_event(kind: str, run_id: Optional[str], tenant_id: Optional[str], payload: Dict[str, Any], conversation_id: Optional[str] = None) -> Dict[str, Any]:
    seq = STORE.next_seq("events")
    evt = {
        "event_id": new_id("evt"),
        "kind": kind,
        "run_id": run_id,
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "payload": payload,
        "created_at": iso(),
        "seq": seq,
    }
    try:
        STORE.execute(
            "INSERT INTO events(event_id, run_id, tenant_id, conversation_id, kind, payload, created_at, seq) VALUES(?,?,?,?,?,?,?,?)",
            (evt["event_id"], run_id, tenant_id, conversation_id, kind, jdump(payload), evt["created_at"], seq),
        )
    except Exception as exc:
        log.warning("event persist failed: %s", exc)
    if run_id:
        BUS.publish(f"run:{run_id}", evt)
    if conversation_id:
        BUS.publish(f"conv:{conversation_id}", evt)
    BUS.publish("global", evt)
    return evt


class ModelClient:
    def __init__(self):
        self.model = MODEL_NAME
        self._client: Optional[OpenAI] = None
        self._lock = threading.RLock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def available(self) -> bool:
        return bool(MODEL_API_KEY)

    def client(self) -> OpenAI:
        with self._lock:
            if self._client is None:
                if not MODEL_API_KEY:
                    raise RuntimeError("MODULAR_API_KEY is not configured")
                self._client = OpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY, timeout=600.0, max_retries=0)
            return self._client

    def _params(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "frequency_penalty": FREQUENCY_PENALTY,
            "presence_penalty": PRESENCE_PENALTY,
            "seed": SEED,
        }
        if override:
            params.update({k: v for k, v in override.items() if v is not None})
        return params

    def stream(
        self,
        messages: List[Dict[str, str]],
        on_delta: Optional[Callable[[str], None]] = None,
        override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = self._params(override)
        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts < 4:
            attempts += 1
            buf: List[str] = []
            usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            try:
                response = self.client().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    **params,
                )
                for chunk in response:
                    if getattr(chunk, "usage", None):
                        u = chunk.usage
                        usage["prompt_tokens"] = int(getattr(u, "prompt_tokens", 0) or 0)
                        usage["completion_tokens"] = int(getattr(u, "completion_tokens", 0) or 0)
                        usage["total_tokens"] = int(getattr(u, "total_tokens", 0) or 0)
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        buf.append(content)
                        if on_delta is not None:
                            on_delta(content)
                text = "".join(buf)
                if usage["total_tokens"] == 0:
                    prompt_text = "\n".join(m.get("content", "") for m in messages)
                    usage["prompt_tokens"] = approx_tokens(prompt_text)
                    usage["completion_tokens"] = approx_tokens(text)
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                with self._lock:
                    self.total_prompt_tokens += usage["prompt_tokens"]
                    self.total_completion_tokens += usage["completion_tokens"]
                return {"text": text, "usage": usage, "attempts": attempts}
            except Exception as exc:
                last_exc = exc
                log.warning("model stream attempt %d failed: %s", attempts, exc)
                time.sleep(min(20.0, 1.5 * (2 ** (attempts - 1))))
        raise RuntimeError(f"model invocation failed after {attempts} attempts: {last_exc}")

    def complete(self, messages: List[Dict[str, str]], override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.stream(messages, on_delta=None, override=override)


MODEL = ModelClient()


class LocalGrammarDecoder:
    OBJ_START = re.compile(r"\{")

    @staticmethod
    def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        candidates: List[str] = list(fenced)
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        candidates.append(text[start : i + 1])
        best: Optional[Dict[str, Any]] = None
        best_score = -1
        for cand in candidates:
            parsed = LocalGrammarDecoder._loose_parse(cand)
            if not isinstance(parsed, dict):
                continue
            score = 0
            for key in ("action", "state_patch", "reasoning", "tool", "arguments", "thought"):
                if key in parsed:
                    score += 2
            score += min(10, len(parsed))
            if score > best_score:
                best_score = score
                best = parsed
        return best

    @staticmethod
    def _loose_parse(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            pass
        cleaned = text
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        cleaned = re.sub(r"//[^\n\r]*", "", cleaned)
        cleaned = cleaned.replace("\t", " ")
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        repaired = cleaned
        open_braces = repaired.count("{") - repaired.count("}")
        if open_braces > 0:
            repaired = repaired + ("}" * open_braces)
        open_brackets = repaired.count("[") - repaired.count("]")
        if open_brackets > 0:
            repaired = repaired + ("]" * open_brackets)
        try:
            return json.loads(repaired)
        except Exception:
            return None


DECODER = LocalGrammarDecoder()


@dataclass
class ProceduralSpec:
    spec_id: str
    tenant_id: str
    objective: str
    success_criteria: List[str]
    constraints: List[str]
    allowed_tools: List[str]
    max_steps: int
    verifiers: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def build(
        tenant_id: str,
        objective: str,
        success_criteria: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        max_steps: int = DEFAULT_STEP_BUDGET,
        verifiers: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ProceduralSpec":
        tools = allowed_tools or TenantRegistry.DEFAULT_TOOLS
        return ProceduralSpec(
            spec_id=new_id("spec"),
            tenant_id=tenant_id,
            objective=objective.strip(),
            success_criteria=[s.strip() for s in (success_criteria or []) if s and s.strip()],
            constraints=[c.strip() for c in (constraints or []) if c and c.strip()],
            allowed_tools=list(tools),
            max_steps=max(1, min(int(max_steps), 5000)),
            verifiers=verifiers or [],
            metadata=metadata or {},
            created_at=iso(),
        )

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProceduralSpec":
        return ProceduralSpec(
            spec_id=d.get("spec_id") or new_id("spec"),
            tenant_id=d.get("tenant_id") or "public",
            objective=d.get("objective") or "",
            success_criteria=list(d.get("success_criteria") or []),
            constraints=list(d.get("constraints") or []),
            allowed_tools=list(d.get("allowed_tools") or TenantRegistry.DEFAULT_TOOLS),
            max_steps=int(d.get("max_steps") or DEFAULT_STEP_BUDGET),
            verifiers=list(d.get("verifiers") or []),
            metadata=dict(d.get("metadata") or {}),
            created_at=d.get("created_at") or iso(),
        )


SIGMA_ALLOWED_KEYS = {
    "phase",
    "progress",
    "open_goals",
    "dependencies",
    "constraints",
    "facts",
    "artifacts",
    "errors",
    "metrics",
    "plan",
    "current_subgoal",
    "skill_notes",
    "verification",
    "cursor",
    "scratch",
}

SIGMA_LIST_KEYS = {"progress", "open_goals", "dependencies", "constraints", "errors", "plan", "skill_notes"}
SIGMA_DICT_KEYS = {"facts", "artifacts", "metrics", "verification", "cursor", "scratch"}
SIGMA_STR_KEYS = {"phase", "current_subgoal"}

SIGMA_LIST_CAP = 64
SIGMA_DICT_CAP = 96
SIGMA_STR_CAP = 4096
SIGMA_TOTAL_CAP = 65536


def empty_sigma() -> Dict[str, Any]:
    return {
        "phase": "init",
        "progress": [],
        "open_goals": [],
        "dependencies": [],
        "constraints": [],
        "facts": {},
        "artifacts": {},
        "errors": [],
        "metrics": {},
        "plan": [],
        "current_subgoal": "",
        "skill_notes": [],
        "verification": {},
        "cursor": {},
        "scratch": {},
    }


class StatePatchValidator:
    DELETE_SENTINELS = {None, "__DELETE__", "$delete", "null"}

    @staticmethod
    def _truncate_str(value: str, cap: int = SIGMA_STR_CAP) -> str:
        v = str(value)
        return v if len(v) <= cap else v[: cap - 3] + "..."

    @classmethod
    def validate_patch(cls, patch: Any) -> Dict[str, Any]:
        if patch is None:
            return {}
        if not isinstance(patch, dict):
            raise ValidationError("state_patch must be an object")
        clean: Dict[str, Any] = {}
        for key, value in patch.items():
            if not isinstance(key, str):
                raise ValidationError("state_patch keys must be strings")
            k = key.strip()
            if not k:
                continue
            if k not in SIGMA_ALLOWED_KEYS:
                clean.setdefault("facts", {})
                if isinstance(clean["facts"], dict):
                    clean["facts"][k[:120]] = cls._coerce_scalar(value)
                continue
            clean[k] = cls._coerce_slot(k, value)
        return clean

    @classmethod
    def _coerce_scalar(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            return cls._truncate_str(value)
        if isinstance(value, (list, tuple)):
            return [cls._coerce_scalar(v) for v in list(value)[:SIGMA_LIST_CAP]]
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= SIGMA_DICT_CAP:
                    break
                out[str(k)[:120]] = cls._coerce_scalar(v)
            return out
        return cls._truncate_str(str(value))

    @classmethod
    def _coerce_slot(cls, key: str, value: Any) -> Any:
        if value in (None,) or (isinstance(value, str) and value in cls.DELETE_SENTINELS and key not in SIGMA_STR_KEYS):
            return None
        if key in SIGMA_STR_KEYS:
            if value is None:
                return None
            return cls._truncate_str(str(value), 1024)
        if key in SIGMA_LIST_KEYS:
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, (list, tuple)):
                raise ValidationError(f"{key} must be a list")
            out: List[Any] = []
            seen: Set[str] = set()
            for item in list(value)[: SIGMA_LIST_CAP * 2]:
                coerced = cls._coerce_scalar(item)
                sig = jdump(coerced)
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(coerced)
                if len(out) >= SIGMA_LIST_CAP:
                    break
            return out
        if key in SIGMA_DICT_KEYS:
            if not isinstance(value, dict):
                raise ValidationError(f"{key} must be an object")
            out_d: Dict[str, Any] = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= SIGMA_DICT_CAP:
                    break
                out_d[str(k)[:120]] = cls._coerce_scalar(v)
            return out_d
        return cls._coerce_scalar(value)

    @classmethod
    def apply(cls, sigma: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        base = json.loads(jdump(sigma)) if sigma else empty_sigma()
        for key in list(base.keys()):
            if key not in SIGMA_ALLOWED_KEYS:
                base.pop(key, None)
        for key, value in patch.items():
            if value is None:
                if key in SIGMA_LIST_KEYS:
                    base[key] = []
                elif key in SIGMA_DICT_KEYS:
                    base[key] = {}
                elif key in SIGMA_STR_KEYS:
                    base[key] = ""
                else:
                    base.pop(key, None)
                continue
            if key in SIGMA_LIST_KEYS:
                existing = base.get(key) or []
                if not isinstance(existing, list):
                    existing = []
                merged: List[Any] = []
                seen: Set[str] = set()
                for item in list(existing) + list(value):
                    sig = jdump(item)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    merged.append(item)
                base[key] = merged[-SIGMA_LIST_CAP:]
            elif key in SIGMA_DICT_KEYS:
                existing_d = base.get(key) or {}
                if not isinstance(existing_d, dict):
                    existing_d = {}
                for k, v in value.items():
                    if v is None:
                        existing_d.pop(k, None)
                    else:
                        existing_d[k] = v
                if len(existing_d) > SIGMA_DICT_CAP:
                    keys = list(existing_d.keys())[-SIGMA_DICT_CAP:]
                    existing_d = {k: existing_d[k] for k in keys}
                base[key] = existing_d
            else:
                base[key] = value
        encoded = jdump(base)
        if len(encoded) > SIGMA_TOTAL_CAP:
            base = cls._compact(base)
        return base

    @classmethod
    def _compact(cls, sigma: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(sigma)
        for key in ("scratch", "skill_notes", "errors"):
            val = out.get(key)
            if isinstance(val, list) and len(val) > 12:
                out[key] = val[-12:]
            elif isinstance(val, dict) and len(val) > 12:
                keys = list(val.keys())[-12:]
                out[key] = {k: val[k] for k in keys}
        for key in ("progress", "plan"):
            val = out.get(key)
            if isinstance(val, list) and len(val) > 24:
                out[key] = val[-24:]
        for key in ("facts", "artifacts", "metrics"):
            val = out.get(key)
            if isinstance(val, dict) and len(val) > 40:
                keys = list(val.keys())[-40:]
                out[key] = {k: val[k] for k in keys}
        encoded = jdump(out)
        if len(encoded) > SIGMA_TOTAL_CAP:
            out["scratch"] = {"compacted_at": iso()}
            out["errors"] = (out.get("errors") or [])[-4:]
        return out


VALIDATOR = StatePatchValidator()


@dataclass
class Observation:
    step: int
    source: str
    tool: Optional[str]
    ok: bool
    summary: str
    data: Dict[str, Any]
    error: Optional[str]
    latency_ms: int
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def initial(step: int, note: str) -> "Observation":
        return Observation(
            step=step,
            source="runtime",
            tool=None,
            ok=True,
            summary=note,
            data={},
            error=None,
            latency_ms=0,
            created_at=iso(),
        )


class WorkspaceManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.RLock] = {}
        self._lock_guard = threading.RLock()

    def _lock_for(self, key: str) -> threading.RLock:
        with self._lock_guard:
            lk = self._locks.get(key)
            if lk is None:
                lk = threading.RLock()
                self._locks[key] = lk
            return lk

    def tenant_root(self, tenant_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tenant_id)[:64] or "unknown"
        p = self.root / safe
        p.mkdir(parents=True, exist_ok=True)
        return p

    def run_root(self, tenant_id: str, run_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)[:64] or "run"
        p = self.tenant_root(tenant_id) / safe
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolve(self, tenant_id: str, run_id: str, rel_path: str) -> Path:
        base = self.run_root(tenant_id, run_id).resolve()
        candidate = (base / (rel_path or "")).resolve()
        if candidate != base and base not in candidate.parents:
            raise SecurityError(f"path escape blocked: {rel_path}")
        return candidate

    def write_file(self, tenant_id: str, run_id: str, rel_path: str, content: str) -> Dict[str, Any]:
        target = self.resolve(tenant_id, run_id, rel_path)
        with self._lock_for(str(target)):
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_", suffix=".part")
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(content)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
            except Exception:
                with contextlib.suppress(Exception):
                    os.unlink(tmp)
                raise
            data = content.encode("utf-8")
            self._index(tenant_id, run_id, target, data)
            return {"path": rel_path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "lines": content.count("\n") + (0 if content.endswith("\n") or not content else 1)}

    def read_file(self, tenant_id: str, run_id: str, rel_path: str, start: int = 1, end: Optional[int] = None) -> Dict[str, Any]:
        target = self.resolve(tenant_id, run_id, rel_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"file not found: {rel_path}")
        with self._lock_for(str(target)):
            text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        s = max(1, int(start))
        e = len(lines) if end is None else max(s, int(end))
        segment = lines[s - 1 : e]
        return {
            "path": rel_path,
            "total_lines": len(lines),
            "start": s,
            "end": min(e, len(lines)),
            "content": "\n".join(segment),
        }

    def append_file(self, tenant_id: str, run_id: str, rel_path: str, lines: Union[str, List[str]], unique: bool = True) -> Dict[str, Any]:
        target = self.resolve(tenant_id, run_id, rel_path)
        incoming = [lines] if isinstance(lines, str) else [str(x) for x in lines]
        normalized: List[str] = []
        for chunk in incoming:
            for part in str(chunk).split("\n"):
                normalized.append(part.rstrip("\r"))
        with self._lock_for(str(target)):
            target.parent.mkdir(parents=True, exist_ok=True)
            existing_text = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
            existing_lines = existing_text.splitlines()
            existing_set = set(existing_lines) if unique else set()
            added: List[str] = []
            skipped: List[str] = []
            for ln in normalized:
                if unique and (ln in existing_set):
                    skipped.append(ln)
                    continue
                added.append(ln)
                if unique:
                    existing_set.add(ln)
            if added:
                body = existing_text
                if body and not body.endswith("\n"):
                    body += "\n"
                body += "\n".join(added) + "\n"
                fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_", suffix=".part")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                        fh.write(body)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, target)
                except Exception:
                    with contextlib.suppress(Exception):
                        os.unlink(tmp)
                    raise
                self._index(tenant_id, run_id, target, body.encode("utf-8"))
            total = len((target.read_text(encoding="utf-8", errors="replace") if target.exists() else "").splitlines())
        return {"path": rel_path, "added": len(added), "skipped": len(skipped), "total_lines": total, "unique": unique}

    def replace_lines(self, tenant_id: str, run_id: str, rel_path: str, start: int, end: int, content: Union[str, List[str]]) -> Dict[str, Any]:
        target = self.resolve(tenant_id, run_id, rel_path)
        if not target.exists():
            raise FileNotFoundError(f"file not found: {rel_path}")
        new_lines = content.split("\n") if isinstance(content, str) else [str(x) for x in content]
        with self._lock_for(str(target)):
            text = target.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            s = max(1, int(start))
            e = max(s - 1, min(int(end), len(lines)))
            if s > len(lines) + 1:
                raise ValidationError(f"start line {s} beyond file length {len(lines)}")
            replaced = lines[s - 1 : e]
            merged = lines[: s - 1] + new_lines + lines[e:]
            body = "\n".join(merged)
            if body and not body.endswith("\n"):
                body += "\n"
            fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_", suffix=".part")
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(body)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
            except Exception:
                with contextlib.suppress(Exception):
                    os.unlink(tmp)
                raise
            self._index(tenant_id, run_id, target, body.encode("utf-8"))
        return {
            "path": rel_path,
            "replaced_lines": len(replaced),
            "inserted_lines": len(new_lines),
            "total_lines": len(merged),
            "start": s,
            "end": e,
        }

    def check_lines(self, tenant_id: str, run_id: str, rel_path: str, candidates: List[str]) -> Dict[str, Any]:
        target = self.resolve(tenant_id, run_id, rel_path)
        present: Dict[str, bool] = {}
        existing: Set[str] = set()
        if target.exists() and target.is_file():
            with self._lock_for(str(target)):
                existing = set(target.read_text(encoding="utf-8", errors="replace").splitlines())
        for cand in candidates:
            present[str(cand)] = str(cand) in existing
        return {"path": rel_path, "results": present, "missing": [k for k, v in present.items() if not v], "found": [k for k, v in present.items() if v]}

    def list_dir(self, tenant_id: str, run_id: str, rel_path: str = ".") -> Dict[str, Any]:
        target = self.resolve(tenant_id, run_id, rel_path)
        if not target.exists():
            return {"path": rel_path, "entries": []}
        entries: List[Dict[str, Any]] = []
        base = self.run_root(tenant_id, run_id)
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
            try:
                stat = child.stat()
            except Exception:
                continue
            entries.append(
                {
                    "name": child.name,
                    "rel_path": str(child.relative_to(base)),
                    "type": "dir" if child.is_dir() else "file",
                    "bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                }
            )
        return {"path": rel_path, "entries": entries[:400]}

    def delete_file(self, tenant_id: str, run_id: str, rel_path: str) -> Dict[str, Any]:
        target = self.resolve(tenant_id, run_id, rel_path)
        base = self.run_root(tenant_id, run_id).resolve()
        if target == base:
            raise SecurityError("cannot delete workspace root")
        with self._lock_for(str(target)):
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
            else:
                return {"path": rel_path, "deleted": False}
        STORE.execute("DELETE FROM file_index WHERE tenant_id = ? AND rel_path = ?", (tenant_id, str(rel_path)))
        return {"path": rel_path, "deleted": True}

    def _index(self, tenant_id: str, run_id: str, target: Path, data: bytes) -> None:
        try:
            rel = str(target.relative_to(self.tenant_root(tenant_id)))
        except Exception:
            rel = target.name
        STORE.execute(
            "INSERT INTO file_index(file_id, tenant_id, run_id, rel_path, sha256, bytes, updated_at) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(tenant_id, rel_path) DO UPDATE SET sha256=excluded.sha256, bytes=excluded.bytes, updated_at=excluded.updated_at, run_id=excluded.run_id",
            (new_id("file"), tenant_id, run_id, rel, hashlib.sha256(data).hexdigest(), len(data), iso()),
        )


WORKSPACE = WorkspaceManager(WORKSPACE_ROOT)


@dataclass
class Skill:
    skill_id: str
    tenant_id: str
    name: str
    version: int
    category: str
    summary: str
    preconditions: List[str]
    procedure: List[str]
    failure_modes: List[str]
    tags: List[str]
    uses: int
    successes: int
    failures: int
    score: float
    status: str

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Skill":
        return Skill(
            skill_id=row["skill_id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            version=int(row["version"]),
            category=row["category"],
            summary=row["summary"],
            preconditions=jload(row["preconditions"], []) or [],
            procedure=jload(row["procedure"], []) or [],
            failure_modes=jload(row["failure_modes"], []) or [],
            tags=jload(row["tags"], []) or [],
            uses=int(row["uses"]),
            successes=int(row["successes"]),
            failures=int(row["failures"]),
            score=float(row["score"]),
            status=row["status"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        parts = [f"SKILL {self.name} (v{self.version}, score={self.score:.2f}, category={self.category})", f"SUMMARY: {self.summary}"]
        if self.preconditions:
            parts.append("PRECONDITIONS: " + "; ".join(str(p) for p in self.preconditions[:8]))
        if self.procedure:
            parts.append("PROCEDURE:")
            for i, step in enumerate(self.procedure[:14], 1):
                parts.append(f"  {i}. {step}")
        if self.failure_modes:
            parts.append("KNOWN FAILURES: " + "; ".join(str(f) for f in self.failure_modes[:6]))
        return "\n".join(parts)


class ExperientialMemory:
    def __init__(self, store: SQLiteStore, embedder: HashingEmbedder):
        self.store = store
        self.embedder = embedder
        self._lock = threading.RLock()

    def _text_of(self, name: str, summary: str, procedure: List[str], tags: List[str], category: str) -> str:
        return " \n".join([name, category, summary, " ".join(str(p) for p in procedure), " ".join(str(t) for t in tags)])

    def upsert(
        self,
        tenant_id: str,
        name: str,
        summary: str,
        procedure: List[str],
        preconditions: Optional[List[str]] = None,
        failure_modes: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        category: str = "general",
        skill_id: Optional[str] = None,
    ) -> Skill:
        preconditions = [str(x) for x in (preconditions or [])][:16]
        failure_modes = [str(x) for x in (failure_modes or [])][:16]
        tags = [str(x).lower() for x in (tags or [])][:16]
        procedure = [str(x) for x in (procedure or [])][:32]
        text = self._text_of(name, summary, procedure, tags, category)
        emb = pack_vector(self.embedder.embed(text))
        with self._lock:
            row = None
            if skill_id:
                row = self.store.one("SELECT * FROM skills WHERE skill_id = ? AND tenant_id = ?", (skill_id, tenant_id))
            if row is None:
                row = self.store.one("SELECT * FROM skills WHERE tenant_id = ? AND name = ?", (tenant_id, name))
            now = iso()
            if row is None:
                sid = skill_id or new_id("skill")
                self.store.execute(
                    "INSERT INTO skills(skill_id, tenant_id, name, version, category, summary, preconditions, procedure, failure_modes, tags, embedding, uses, successes, failures, score, status, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,0,0,0,?,'active',?,?)",
                    (
                        sid,
                        tenant_id,
                        name,
                        1,
                        category,
                        summary,
                        jdump(preconditions),
                        jdump(procedure),
                        jdump(failure_modes),
                        jdump(tags),
                        emb,
                        0.5,
                        now,
                        now,
                    ),
                )
            else:
                sid = row["skill_id"]
                self.store.execute(
                    "UPDATE skills SET name=?, version=version+1, category=?, summary=?, preconditions=?, procedure=?, failure_modes=?, tags=?, embedding=?, updated_at=?, status='active' WHERE skill_id=?",
                    (
                        name,
                        category,
                        summary,
                        jdump(preconditions),
                        jdump(procedure),
                        jdump(failure_modes),
                        jdump(tags),
                        emb,
                        now,
                        sid,
                    ),
                )
            self.store.execute("DELETE FROM skills_fts WHERE skill_id = ?", (sid,))
            self.store.execute(
                "INSERT INTO skills_fts(skill_id, tenant_id, name, summary, procedure, tags) VALUES(?,?,?,?,?,?)",
                (sid, tenant_id, name, summary, " \n".join(procedure), " ".join(tags)),
            )
            fresh = self.store.one("SELECT * FROM skills WHERE skill_id = ?", (sid,))
            return Skill.from_row(fresh)

    def get(self, tenant_id: str, skill_id: str) -> Optional[Skill]:
        row = self.store.one("SELECT * FROM skills WHERE skill_id = ? AND tenant_id = ?", (skill_id, tenant_id))
        return Skill.from_row(row) if row else None

    def list(self, tenant_id: str, limit: int = 200) -> List[Skill]:
        rows = self.store.query(
            "SELECT * FROM skills WHERE tenant_id = ? AND status = 'active' ORDER BY score DESC, updated_at DESC LIMIT ?",
            (tenant_id, int(limit)),
        )
        return [Skill.from_row(r) for r in rows]

    def retire(self, tenant_id: str, skill_id: str) -> bool:
        self.store.execute("UPDATE skills SET status='retired', updated_at=? WHERE skill_id=? AND tenant_id=?", (iso(), skill_id, tenant_id))
        return True

    def search(self, tenant_id: str, query_text: str, limit: int = 2) -> List[Tuple[Skill, float]]:
        query_text = (query_text or "").strip()
        if not query_text:
            return [(s, s.score) for s in self.list(tenant_id, limit)][:limit]
        rows = self.store.query("SELECT * FROM skills WHERE tenant_id = ? AND status='active'", (tenant_id,))
        if not rows:
            return []
        skills = {r["skill_id"]: r for r in rows}
        qvec = self.embedder.embed(query_text)
        dense_scores: List[Tuple[str, float]] = []
        for sid, r in skills.items():
            sim = self.embedder.cosine(qvec, unpack_vector(r["embedding"]))
            dense_scores.append((sid, sim))
        dense_scores.sort(key=lambda kv: kv[1], reverse=True)
        dense_ranking = [sid for sid, _ in dense_scores[:50]]
        sparse_ranking: List[str] = []
        match = fts_escape(query_text)
        if match:
            try:
                frows = self.store.query(
                    "SELECT skill_id, bm25(skills_fts) AS rank FROM skills_fts WHERE skills_fts MATCH ? AND tenant_id = ? ORDER BY rank LIMIT 50",
                    (match, tenant_id),
                )
                sparse_ranking = [r["skill_id"] for r in frows if r["skill_id"] in skills]
            except Exception as exc:
                log.debug("skills fts failed: %s", exc)
        prior_ranking = [sid for sid, _ in sorted(((k, float(v["score"])) for k, v in skills.items()), key=lambda kv: kv[1], reverse=True)[:50]]
        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking, prior_ranking], weights=[1.0, 0.85, 0.35])
        out: List[Tuple[Skill, float]] = []
        for sid, score in fused[: max(1, limit)]:
            row = skills.get(sid)
            if row is None:
                continue
            out.append((Skill.from_row(row), float(score)))
        return out

    def record_outcome(self, skill_id: str, success: bool) -> None:
        with self.store.tx() as c:
            row = c.execute("SELECT uses, successes, failures FROM skills WHERE skill_id = ?", (skill_id,)).fetchone()
            if row is None:
                return
            uses = int(row[0]) + 1
            successes = int(row[1]) + (1 if success else 0)
            failures = int(row[2]) + (0 if success else 1)
            score = (successes + 1.0) / (uses + 2.0)
            c.execute(
                "UPDATE skills SET uses=?, successes=?, failures=?, score=?, updated_at=? WHERE skill_id=?",
                (uses, successes, failures, score, iso(), skill_id),
            )


EM = ExperientialMemory(STORE, EMBEDDER)


class WorkingMemory:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def load(self, tenant_id: str, run_id: str) -> Dict[str, Any]:
        row = self.store.one("SELECT * FROM working_memory WHERE run_id = ? AND tenant_id = ?", (run_id, tenant_id))
        if row is None:
            return {"progress": [], "open_goals": [], "dependencies": [], "constraints": [], "facts": {}}
        return {
            "progress": jload(row["progress"], []) or [],
            "open_goals": jload(row["open_goals"], []) or [],
            "dependencies": jload(row["dependencies"], []) or [],
            "constraints": jload(row["constraints"], []) or [],
            "facts": jload(row["facts"], {}) or {},
        }

    def sync_from_sigma(self, tenant_id: str, run_id: str, sigma: Dict[str, Any]) -> Dict[str, Any]:
        wm = {
            "progress": list(sigma.get("progress") or [])[-24:],
            "open_goals": list(sigma.get("open_goals") or [])[-24:],
            "dependencies": list(sigma.get("dependencies") or [])[-24:],
            "constraints": list(sigma.get("constraints") or [])[-24:],
            "facts": dict(sigma.get("facts") or {}),
        }
        row = self.store.one("SELECT wm_id FROM working_memory WHERE run_id = ?", (run_id,))
        now = iso()
        if row is None:
            self.store.execute(
                "INSERT INTO working_memory(wm_id, run_id, tenant_id, progress, open_goals, dependencies, constraints, facts, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (new_id("wm"), run_id, tenant_id, jdump(wm["progress"]), jdump(wm["open_goals"]), jdump(wm["dependencies"]), jdump(wm["constraints"]), jdump(wm["facts"]), now),
            )
        else:
            self.store.execute(
                "UPDATE working_memory SET progress=?, open_goals=?, dependencies=?, constraints=?, facts=?, updated_at=? WHERE run_id=?",
                (jdump(wm["progress"]), jdump(wm["open_goals"]), jdump(wm["dependencies"]), jdump(wm["constraints"]), jdump(wm["facts"]), now, run_id),
            )
        return wm

    def routing_query(self, spec: ProceduralSpec, sigma: Dict[str, Any], observation: Observation) -> str:
        pieces: List[str] = [spec.objective]
        subgoal = sigma.get("current_subgoal") or ""
        if subgoal:
            pieces.append(str(subgoal))
        for goal in list(sigma.get("open_goals") or [])[:4]:
            pieces.append(str(goal))
        for err in list(sigma.get("errors") or [])[-3:]:
            pieces.append(str(err))
        if observation and observation.summary:
            pieces.append(observation.summary[:400])
        if observation and observation.error:
            pieces.append(observation.error[:400])
        for c in list(sigma.get("constraints") or [])[:3]:
            pieces.append(str(c))
        return "\n".join(p for p in pieces if p)[:4000]


WM = WorkingMemory(STORE)


class WikiKnowledgeBase:
    def __init__(self, store: SQLiteStore, root: Path, embedder: HashingEmbedder):
        self.store = store
        self.root = root
        self.embedder = embedder
        self.root.mkdir(parents=True, exist_ok=True)
        self._git_ready = self._init_git()
        self._lock = threading.RLock()

    def _git(self, *args: str) -> Tuple[int, str]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except FileNotFoundError:
            return 127, "git not installed"
        except Exception as exc:
            return 1, str(exc)

    def _init_git(self) -> bool:
        if shutil.which("git") is None:
            log.info("git unavailable; wiki versioning will use internal revisions only")
            return False
        if not (self.root / ".git").exists():
            code, out = self._git("init")
            if code != 0:
                log.warning("git init failed: %s", out)
                return False
            self._git("config", "user.email", "agent@runtime.local")
            self._git("config", "user.name", "Agent Runtime")
        return True

    @staticmethod
    def slugify(title: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "-", (title or "page").lower()).strip("-")
        return (s or "page")[:80]

    @staticmethod
    def unified_diff(old: str, new: str) -> str:
        import difflib

        diff = difflib.unified_diff(
            old.splitlines(keepends=False),
            new.splitlines(keepends=False),
            fromfile="previous",
            tofile="current",
            lineterm="",
            n=2,
        )
        return "\n".join(list(diff)[:600])

    def upsert(self, tenant_id: str, title: str, body: str, category: str = "general") -> Dict[str, Any]:
        slug = self.slugify(title)
        with self._lock:
            row = self.store.one("SELECT * FROM wiki_pages WHERE tenant_id = ? AND slug = ?", (tenant_id, slug))
            now = iso()
            emb = pack_vector(self.embedder.embed(f"{title}\n{category}\n{body}"))
            if row is None:
                page_id = new_id("wiki")
                version = 1
                old_body = ""
                self.store.execute(
                    "INSERT INTO wiki_pages(page_id, tenant_id, slug, title, category, body, version, embedding, updated_at, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (page_id, tenant_id, slug, title, category, body, version, emb, now, now),
                )
            else:
                page_id = row["page_id"]
                version = int(row["version"]) + 1
                old_body = row["body"]
                self.store.execute(
                    "UPDATE wiki_pages SET title=?, category=?, body=?, version=?, embedding=?, updated_at=? WHERE page_id=?",
                    (title, category, body, version, emb, now, page_id),
                )
            diff = self.unified_diff(old_body, body)
            self.store.execute(
                "INSERT INTO wiki_revisions(revision_id, page_id, tenant_id, version, diff, body, created_at) VALUES(?,?,?,?,?,?,?)",
                (new_id("rev"), page_id, tenant_id, version, diff, body, now),
            )
            self.store.execute("DELETE FROM wiki_fts WHERE page_id = ?", (page_id,))
            self.store.execute(
                "INSERT INTO wiki_fts(page_id, tenant_id, title, body, category) VALUES(?,?,?,?,?)",
                (page_id, tenant_id, title, body, category),
            )
            tenant_dir = self.root / re.sub(r"[^A-Za-z0-9_.-]", "_", tenant_id)[:64]
            tenant_dir.mkdir(parents=True, exist_ok=True)
            file_path = tenant_dir / f"{slug}.md"
            header = f"# {title}\n\nCategory: {category}\nVersion: {version}\nUpdated: {now}\n\n"
            file_path.write_text(header + body + "\n", encoding="utf-8")
            commit_out = ""
            if self._git_ready:
                self._git("add", "-A")
                code, commit_out = self._git("commit", "-m", f"wiki: {tenant_id}/{slug} v{version}")
                if code != 0 and "nothing to commit" not in commit_out:
                    log.debug("git commit note: %s", commit_out[:200])
            return {"page_id": page_id, "slug": slug, "version": version, "diff": diff, "path": str(file_path.relative_to(self.root))}

    def search(self, tenant_id: str, query_text: str, limit: int = 4) -> List[Dict[str, Any]]:
        rows = self.store.query("SELECT * FROM wiki_pages WHERE tenant_id = ?", (tenant_id,))
        if not rows:
            return []
        by_id = {r["page_id"]: r for r in rows}
        qvec = self.embedder.embed(query_text or "")
        dense = sorted(((r["page_id"], self.embedder.cosine(qvec, unpack_vector(r["embedding"]))) for r in rows), key=lambda kv: kv[1], reverse=True)
        dense_ranking = [pid for pid, _ in dense[:50]]
        sparse_ranking: List[str] = []
        match = fts_escape(query_text or "")
        if match:
            try:
                frows = self.store.query(
                    "SELECT page_id, bm25(wiki_fts) AS rank FROM wiki_fts WHERE wiki_fts MATCH ? AND tenant_id = ? ORDER BY rank LIMIT 50",
                    (match, tenant_id),
                )
                sparse_ranking = [r["page_id"] for r in frows if r["page_id"] in by_id]
            except Exception as exc:
                log.debug("wiki fts failed: %s", exc)
        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], weights=[1.0, 1.0])
        out: List[Dict[str, Any]] = []
        for pid, score in fused[: max(1, limit)]:
            r = by_id.get(pid)
            if r is None:
                continue
            out.append(
                {
                    "page_id": pid,
                    "slug": r["slug"],
                    "title": r["title"],
                    "category": r["category"],
                    "version": int(r["version"]),
                    "score": float(score),
                    "excerpt": (r["body"] or "")[:900],
                }
            )
        return out

    def list_pages(self, tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.store.query(
            "SELECT page_id, slug, title, category, version, updated_at FROM wiki_pages WHERE tenant_id = ? ORDER BY updated_at DESC LIMIT ?",
            (tenant_id, int(limit)),
        )
        return [dict(r) for r in rows]

    def get_page(self, tenant_id: str, slug: str) -> Optional[Dict[str, Any]]:
        row = self.store.one("SELECT * FROM wiki_pages WHERE tenant_id = ? AND slug = ?", (tenant_id, slug))
        if row is None:
            return None
        return {
            "page_id": row["page_id"],
            "slug": row["slug"],
            "title": row["title"],
            "category": row["category"],
            "version": int(row["version"]),
            "body": row["body"],
            "updated_at": row["updated_at"],
        }


WIKI = WikiKnowledgeBase(STORE, WIKI_ROOT, EMBEDDER)


class RawTraceLayer:
    def __init__(self, store: SQLiteStore, root: Path):
        self.store = store
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def append(
        self,
        tenant_id: str,
        run_id: str,
        step: int,
        kind: str,
        pre_state: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        state_delta: Dict[str, Any],
        success: bool,
        latency_ms: int,
        skill_id: Optional[str] = None,
        receipt: Optional[Dict[str, Any]] = None,
    ) -> str:
        trace_id = new_id("trc")
        payload = {
            "trace_id": trace_id,
            "tenant_id": tenant_id,
            "run_id": run_id,
            "step": step,
            "kind": kind,
            "skill_id": skill_id,
            "pre_state": pre_state,
            "action": action,
            "outcome": outcome,
            "state_delta": state_delta,
            "success": bool(success),
            "latency_ms": int(latency_ms),
            "created_at": iso(),
        }
        receipt = receipt or {}
        receipt.setdefault("signature", sign_payload(payload))
        receipt.setdefault("host", os.environ.get("HOSTNAME", "local"))
        receipt.setdefault("pid", os.getpid())
        digest = stable_hash(payload)
        self.store.execute(
            "INSERT INTO raw_traces(trace_id, run_id, tenant_id, step, kind, skill_id, pre_state, action, outcome, state_delta, receipt, success, latency_ms, created_at, digest) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trace_id,
                run_id,
                tenant_id,
                step,
                kind,
                skill_id,
                jdump(pre_state),
                jdump(action),
                jdump(outcome),
                jdump(state_delta),
                jdump(receipt),
                1 if success else 0,
                int(latency_ms),
                payload["created_at"],
                digest,
            ),
        )
        with self._lock:
            path = self.root / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', run_id)}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(jdump({**payload, "receipt": receipt, "digest": digest}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return trace_id

    def for_run(self, run_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        rows = self.store.query("SELECT * FROM raw_traces WHERE run_id = ? ORDER BY step ASC, created_at ASC LIMIT ?", (run_id, int(limit)))
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "trace_id": r["trace_id"],
                    "step": int(r["step"]),
                    "kind": r["kind"],
                    "skill_id": r["skill_id"],
                    "action": jload(r["action"], {}),
                    "outcome": jload(r["outcome"], {}),
                    "state_delta": jload(r["state_delta"], {}),
                    "success": bool(r["success"]),
                    "latency_ms": int(r["latency_ms"]),
                    "created_at": r["created_at"],
                }
            )
        return out

    def failures(self, tenant_id: str, limit: int = 60) -> List[Dict[str, Any]]:
        rows = self.store.query(
            "SELECT * FROM raw_traces WHERE tenant_id = ? AND success = 0 ORDER BY created_at DESC LIMIT ?",
            (tenant_id, int(limit)),
        )
        return [
            {
                "trace_id": r["trace_id"],
                "run_id": r["run_id"],
                "step": int(r["step"]),
                "kind": r["kind"],
                "skill_id": r["skill_id"],
                "action": jload(r["action"], {}),
                "outcome": jload(r["outcome"], {}),
                "created_at": r["created_at"],
            }
            for r in rows
        ]


TRACES = RawTraceLayer(STORE, TRACE_ROOT)


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
}

BLOCKED_PY_PATTERNS = [
    r"\bimport\s+(os|sys|subprocess|socket|shutil|ctypes|multiprocessing|threading|pickle|marshal|importlib|pty|signal|resource)\b",
    r"\bfrom\s+(os|sys|subprocess|socket|shutil|ctypes|multiprocessing|threading|pickle|marshal|importlib|pty|signal|resource)\s+import\b",
    r"__import__",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\bopen\s*\(",
    r"\bglobals\s*\(",
    r"\blocals\s*\(",
    r"\bvars\s*\(",
    r"\bgetattr\s*\(",
    r"\bsetattr\s*\(",
    r"\bdelattr\s*\(",
    r"__subclasses__",
    r"__mro__",
    r"__bases__",
    r"__globals__",
    r"__code__",
    r"__builtins__",
]

SHELL_ALLOWED = {
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "find",
    "sort",
    "uniq",
    "cut",
    "tr",
    "sed",
    "awk",
    "diff",
    "echo",
    "pwd",
    "date",
    "stat",
    "du",
    "mkdir",
    "touch",
    "cp",
    "mv",
    "python3",
    "python",
    "node",
    "jq",
}

SHELL_FORBIDDEN_TOKENS = ["rm ", "rm\t", ":(){", "mkfs", "dd ", "shutdown", "reboot", "chmod 777 /", "chown", "sudo", "curl ", "wget ", "nc ", "ssh ", "/etc/passwd", "/dev/sd"]


class PythonSandbox:
    def __init__(self, timeout_s: float = 20.0, max_output: int = 20000):
        self.timeout_s = timeout_s
        self.max_output = max_output

    def _screen(self, code: str) -> None:
        for pattern in BLOCKED_PY_PATTERNS:
            if re.search(pattern, code):
                raise SecurityError(f"blocked construct matched: {pattern}")

    def run(self, code: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not code or not code.strip():
            raise ValidationError("empty code")
        if len(code) > 200000:
            raise ValidationError("code too large")
        self._screen(code)
        import io

        allowed_modules = {
            "math": __import__("math"),
            "json": __import__("json"),
            "re": __import__("re"),
            "random": __import__("random"),
            "statistics": __import__("statistics"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "collections": __import__("collections"),
            "datetime": __import__("datetime"),
            "decimal": __import__("decimal"),
            "fractions": __import__("fractions"),
            "hashlib": __import__("hashlib"),
            "base64": __import__("base64"),
            "textwrap": __import__("textwrap"),
            "string": __import__("string"),
            "heapq": __import__("heapq"),
            "bisect": __import__("bisect"),
            "difflib": __import__("difflib"),
            "unicodedata": __import__("unicodedata"),
            "uuid": __import__("uuid"),
        }

        def guarded_import(name: str, globals_=None, locals_=None, fromlist=(), level=0):
            root = name.split(".")[0]
            if root not in allowed_modules:
                raise SecurityError(f"import of module '{name}' is not permitted")
            return allowed_modules[root]

        builtins_map = dict(SAFE_BUILTINS)
        builtins_map["__import__"] = guarded_import
        env: Dict[str, Any] = {"__builtins__": builtins_map, "__name__": "sandbox"}
        env.update(allowed_modules)
        if variables:
            for k, v in variables.items():
                if isinstance(k, str) and k.isidentifier():
                    env[k] = v

        stdout = io.StringIO()
        stderr = io.StringIO()
        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, str] = {}

        def target():
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    compiled = compile(code, "<sandbox>", "exec")
                    exec(compiled, env, env)
                    if "result" in env:
                        try:
                            result_holder["result"] = json.loads(jdump(env["result"]))
                        except Exception:
                            result_holder["result"] = str(env["result"])[: self.max_output]
            except BaseException as exc:
                error_holder["error"] = f"{type(exc).__name__}: {exc}"
                error_holder["traceback"] = "".join(traceback.format_exception_only(type(exc), exc))

        thread = threading.Thread(target=target, daemon=True)
        started = time.time()
        thread.start()
        thread.join(self.timeout_s)
        timed_out = thread.is_alive()
        elapsed = int((time.time() - started) * 1000)
        out = stdout.getvalue()[: self.max_output]
        err = stderr.getvalue()[: self.max_output]
        return {
            "ok": (not timed_out) and ("error" not in error_holder),
            "stdout": out,
            "stderr": err,
            "result": result_holder.get("result"),
            "error": error_holder.get("error") if not timed_out else f"timeout after {self.timeout_s}s",
            "latency_ms": elapsed,
            "timed_out": timed_out,
        }


PY_SANDBOX = PythonSandbox()


class ShellExecutor:
    def __init__(self, timeout_s: float = 45.0, max_output: int = 40000):
        self.timeout_s = timeout_s
        self.max_output = max_output

    def _screen(self, command: str) -> List[str]:
        if not command or not command.strip():
            raise ValidationError("empty command")
        low = command.lower()
        for tok in SHELL_FORBIDDEN_TOKENS:
            if tok in low:
                raise SecurityError(f"forbidden shell token: {tok.strip()}")
        if any(ch in command for ch in ("`", "$(", ">", "<", "&")):
            raise SecurityError("shell metacharacters are not permitted")
        import shlex

        segments = [seg.strip() for seg in command.split("|")]
        parsed: List[List[str]] = []
        for seg in segments:
            if not seg:
                raise ValidationError("empty pipeline segment")
            argv = shlex.split(seg)
            if not argv:
                raise ValidationError("empty pipeline segment")
            binary = os.path.basename(argv[0])
            if binary not in SHELL_ALLOWED:
                raise SecurityError(f"binary '{binary}' is not on the allowlist")
            parsed.append(argv)
        if len(parsed) > 1:
            raise SecurityError("pipelines are not permitted")
        return parsed[0]

    def run(self, tenant_id: str, run_id: str, command: str) -> Dict[str, Any]:
        argv = self._screen(command)
        cwd = WORKSPACE.run_root(tenant_id, run_id)
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(cwd),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        started = time.time()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": (proc.stdout or "")[: self.max_output],
                "stderr": (proc.stderr or "")[: self.max_output],
                "latency_ms": int((time.time() - started) * 1000),
                "argv": argv,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"timeout after {self.timeout_s}s",
                "latency_ms": int((time.time() - started) * 1000),
                "argv": argv,
            }


SHELL = ShellExecutor()


class HttpFetcher:
    BLOCKED_HOST_PATTERNS = [
        r"^localhost$",
        r"^127\.",
        r"^0\.",
        r"^10\.",
        r"^192\.168\.",
        r"^172\.(1[6-9]|2[0-9]|3[01])\.",
        r"^169\.254\.",
        r"^::1$",
        r"^fc00:",
        r"^fe80:",
        r"metadata",
    ]

    def __init__(self, timeout_s: float = 25.0, max_bytes: int = 400000):
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes

    def _validate(self, url: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise SecurityError("only http/https schemes are permitted")
        host = (parsed.hostname or "").lower()
        if not host:
            raise SecurityError("missing host")
        for pattern in self.BLOCKED_HOST_PATTERNS:
            if re.search(pattern, host):
                raise SecurityError(f"host '{host}' is blocked by egress policy")
        return url

    def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        import urllib.request
        import urllib.error

        safe_url = self._validate(url)
        req = urllib.request.Request(safe_url, method="GET")
        req.add_header("User-Agent", "AgentRuntime/1.0")
        req.add_header("Accept", "text/plain, text/html, application/json;q=0.9, */*;q=0.5")
        if headers:
            for k, v in list(headers.items())[:12]:
                if str(k).lower() in ("authorization", "cookie", "proxy-authorization"):
                    continue
                req.add_header(str(k)[:64], str(v)[:512])
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read(self.max_bytes + 1)
                truncated = len(raw) > self.max_bytes
                body = raw[: self.max_bytes].decode("utf-8", errors="replace")
                text = re.sub(r"<script[^>]*>.*?</script>", " ", body, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return {
                    "ok": True,
                    "status": resp.status,
                    "url": safe_url,
                    "content_type": resp.headers.get("Content-Type", ""),
                    "text": text[:120000],
                    "truncated": truncated,
                    "latency_ms": int((time.time() - started) * 1000),
                }
        except urllib.error.HTTPError as exc:
            return {"ok": False, "status": exc.code, "url": safe_url, "error": f"HTTP {exc.code}", "latency_ms": int((time.time() - started) * 1000)}
        except Exception as exc:
            return {"ok": False, "status": 0, "url": safe_url, "error": str(exc)[:500], "latency_ms": int((time.time() - started) * 1000)}


HTTP = HttpFetcher()


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]
    mutating: bool = False


class ToolRegistry:
    def __init__(self):
        self.tools: "OrderedDict[str, ToolSpec]" = OrderedDict()

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        spec = self.tools.get(name)
        if spec is None:
            raise ToolDenied(f"unknown tool '{name}'")
        return spec

    def describe(self, allowed: Optional[List[str]] = None) -> str:
        lines: List[str] = []
        for name, spec in self.tools.items():
            if allowed is not None and name not in allowed:
                continue
            args = ", ".join(f"{k}:{v}" for k, v in spec.schema.items())
            lines.append(f"- {name}({args}) :: {spec.description}")
        return "\n".join(lines)

    def names(self) -> List[str]:
        return list(self.tools.keys())


TOOLS = ToolRegistry()


def _tool_write_file(ctx: Dict[str, Any], path: str = "", content: str = "", **_: Any) -> Dict[str, Any]:
    return WORKSPACE.write_file(ctx["tenant_id"], ctx["run_id"], str(path), str(content))


def _tool_read_file(ctx: Dict[str, Any], path: str = "", start: int = 1, end: Optional[int] = None, **_: Any) -> Dict[str, Any]:
    return WORKSPACE.read_file(ctx["tenant_id"], ctx["run_id"], str(path), int(start or 1), int(end) if end is not None else None)


def _tool_append_file(ctx: Dict[str, Any], path: str = "", lines: Any = "", unique: bool = True, **_: Any) -> Dict[str, Any]:
    return WORKSPACE.append_file(ctx["tenant_id"], ctx["run_id"], str(path), lines, bool(unique))


def _tool_replace_lines(ctx: Dict[str, Any], path: str = "", start: int = 1, end: int = 1, content: Any = "", **_: Any) -> Dict[str, Any]:
    return WORKSPACE.replace_lines(ctx["tenant_id"], ctx["run_id"], str(path), int(start), int(end), content)


def _tool_check_lines(ctx: Dict[str, Any], path: str = "", lines: Any = None, **_: Any) -> Dict[str, Any]:
    cands = lines if isinstance(lines, list) else ([lines] if lines else [])
    return WORKSPACE.check_lines(ctx["tenant_id"], ctx["run_id"], str(path), [str(c) for c in cands])


def _tool_list_dir(ctx: Dict[str, Any], path: str = ".", **_: Any) -> Dict[str, Any]:
    return WORKSPACE.list_dir(ctx["tenant_id"], ctx["run_id"], str(path or "."))


def _tool_delete_file(ctx: Dict[str, Any], path: str = "", **_: Any) -> Dict[str, Any]:
    return WORKSPACE.delete_file(ctx["tenant_id"], ctx["run_id"], str(path))


def _tool_wiki_search(ctx: Dict[str, Any], query: str = "", limit: int = 4, **_: Any) -> Dict[str, Any]:
    results = WIKI.search(ctx["tenant_id"], str(query), int(limit or 4))
    return {"query": query, "results": results, "count": len(results)}


def _tool_wiki_write(ctx: Dict[str, Any], title: str = "", body: str = "", category: str = "general", **_: Any) -> Dict[str, Any]:
    if not title or not body:
        raise ValidationError("wiki_write requires title and body")
    return WIKI.upsert(ctx["tenant_id"], str(title), str(body), str(category or "general"))


def _tool_skill_search(ctx: Dict[str, Any], query: str = "", limit: int = 3, **_: Any) -> Dict[str, Any]:
    found = EM.search(ctx["tenant_id"], str(query), int(limit or 3))
    return {"query": query, "results": [{"skill": s.to_dict(), "score": sc} for s, sc in found]}


def _tool_skill_upsert(
    ctx: Dict[str, Any],
    name: str = "",
    summary: str = "",
    procedure: Any = None,
    preconditions: Any = None,
    failure_modes: Any = None,
    tags: Any = None,
    category: str = "general",
    **_: Any,
) -> Dict[str, Any]:
    if not name or not summary:
        raise ValidationError("skill_upsert requires name and summary")
    proc = procedure if isinstance(procedure, list) else ([str(procedure)] if procedure else [])
    pre = preconditions if isinstance(preconditions, list) else ([str(preconditions)] if preconditions else [])
    fails = failure_modes if isinstance(failure_modes, list) else ([str(failure_modes)] if failure_modes else [])
    tg = tags if isinstance(tags, list) else ([str(tags)] if tags else [])
    skill = EM.upsert(ctx["tenant_id"], str(name), str(summary), proc, pre, fails, tg, str(category or "general"))
    return {"skill_id": skill.skill_id, "name": skill.name, "version": skill.version}


def _tool_trace_query(ctx: Dict[str, Any], scope: str = "run", limit: int = 20, **_: Any) -> Dict[str, Any]:
    if scope == "failures":
        return {"scope": scope, "traces": TRACES.failures(ctx["tenant_id"], int(limit or 20))}
    return {"scope": "run", "traces": TRACES.for_run(ctx["run_id"], int(limit or 20))}


def _tool_python(ctx: Dict[str, Any], code: str = "", variables: Any = None, **_: Any) -> Dict[str, Any]:
    vars_map = variables if isinstance(variables, dict) else {}
    return PY_SANDBOX.run(str(code), vars_map)


def _tool_shell(ctx: Dict[str, Any], command: str = "", **_: Any) -> Dict[str, Any]:
    return SHELL.run(ctx["tenant_id"], ctx["run_id"], str(command))


def _tool_http_get(ctx: Dict[str, Any], url: str = "", headers: Any = None, **_: Any) -> Dict[str, Any]:
    hdrs = headers if isinstance(headers, dict) else None
    return HTTP.get(str(url), hdrs)


def _tool_think(ctx: Dict[str, Any], note: str = "", **_: Any) -> Dict[str, Any]:
    return {"acknowledged": True, "note": str(note)[:2000]}


def _tool_finish(ctx: Dict[str, Any], summary: str = "", artifacts: Any = None, **_: Any) -> Dict[str, Any]:
    return {"terminal": True, "status": "completed", "summary": str(summary)[:8000], "artifacts": artifacts if isinstance(artifacts, dict) else {}}


def _tool_fail(ctx: Dict[str, Any], reason: str = "", **_: Any) -> Dict[str, Any]:
    return {"terminal": True, "status": "failed", "reason": str(reason)[:4000]}


TOOLS.register(ToolSpec("workspace.write_file", "Atomically write a UTF-8 text file in the run workspace.", {"path": "string", "content": "string"}, _tool_write_file, True))
TOOLS.register(ToolSpec("workspace.read_file", "Read a file or line range from the run workspace.", {"path": "string", "start": "int?", "end": "int?"}, _tool_read_file))
TOOLS.register(ToolSpec("workspace.append_file", "Append lines with optional exact-line deduplication.", {"path": "string", "lines": "string|string[]", "unique": "bool"}, _tool_append_file, True))
TOOLS.register(ToolSpec("workspace.replace_lines", "Replace an inclusive 1-indexed line range with new content.", {"path": "string", "start": "int", "end": "int", "content": "string|string[]"}, _tool_replace_lines, True))
TOOLS.register(ToolSpec("workspace.check_lines", "Batched exact-line membership test against a file.", {"path": "string", "lines": "string[]"}, _tool_check_lines))
TOOLS.register(ToolSpec("workspace.list_dir", "List entries of a workspace directory.", {"path": "string"}, _tool_list_dir))
TOOLS.register(ToolSpec("workspace.delete_file", "Delete a workspace file or directory.", {"path": "string"}, _tool_delete_file, True))
TOOLS.register(ToolSpec("memory.wiki_search", "Hybrid dense+BM25 RRF search over the persistent knowledge wiki.", {"query": "string", "limit": "int"}, _tool_wiki_search))
TOOLS.register(ToolSpec("memory.wiki_write", "Create or update a versioned markdown wiki page.", {"title": "string", "body": "string", "category": "string"}, _tool_wiki_write, True))
TOOLS.register(ToolSpec("memory.skill_search", "Search the experiential skill library.", {"query": "string", "limit": "int"}, _tool_skill_search))
TOOLS.register(ToolSpec("memory.skill_upsert", "Create or revise a reusable procedural skill.", {"name": "string", "summary": "string", "procedure": "string[]", "preconditions": "string[]", "failure_modes": "string[]", "tags": "string[]", "category": "string"}, _tool_skill_upsert, True))
TOOLS.register(ToolSpec("memory.trace_query", "Query immutable execution traces ('run' or 'failures').", {"scope": "string", "limit": "int"}, _tool_trace_query))
TOOLS.register(ToolSpec("compute.python", "Execute sandboxed pure-Python computation; assign 'result' to return data.", {"code": "string", "variables": "object"}, _tool_python))
TOOLS.register(ToolSpec("compute.shell", "Run a single allowlisted binary inside the run workspace.", {"command": "string"}, _tool_shell, True))
TOOLS.register(ToolSpec("compute.http_get", "Fetch a public http(s) URL and return extracted text.", {"url": "string", "headers": "object"}, _tool_http_get))
TOOLS.register(ToolSpec("reason.think", "Record a within-step deliberation note without side effects.", {"note": "string"}, _tool_think))
TOOLS.register(ToolSpec("control.finish", "Terminate the run as completed with a final summary.", {"summary": "string", "artifacts": "object"}, _tool_finish))
TOOLS.register(ToolSpec("control.fail", "Terminate the run as failed with a reason.", {"reason": "string"}, _tool_fail))


class ZeroTrustGate:
    MUTATING_TOOLS = {name for name, spec in TOOLS.tools.items() if spec.mutating}

    @staticmethod
    def verify(tenant_id: str, run_id: str, spec: ProceduralSpec, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool not in TOOLS.names():
            raise ToolDenied(f"tool '{tool}' is not registered")
        if tool not in spec.allowed_tools:
            raise ToolDenied(f"tool '{tool}' is not permitted by the procedural specification")
        TENANTS.authorize_tool(tenant_id, tool)
        if not isinstance(arguments, dict):
            raise ValidationError("tool arguments must be an object")
        encoded = jdump(arguments)
        if len(encoded) > 400000:
            raise ValidationError("tool arguments payload too large")
        path = arguments.get("path")
        if isinstance(path, str):
            if path.startswith("/") or ".." in Path(path).parts:
                raise SecurityError("absolute or traversal paths are prohibited")
        receipt = {
            "tool": tool,
            "tenant_id": tenant_id,
            "run_id": run_id,
            "arg_digest": stable_hash(arguments),
            "mutating": tool in ZeroTrustGate.MUTATING_TOOLS,
            "authorized_at": iso(),
        }
        receipt["signature"] = sign_payload(receipt)
        audit(tenant_id, run_id, "runtime", f"tool_authorized:{tool}", {"arg_digest": receipt["arg_digest"]}, True)
        return receipt


class OutputClassifier:
    SECRET_PATTERNS = [
        (re.compile(r"(?i)\b(sk|pk)-[A-Za-z0-9]{16,}\b"), "api_key"),
        (re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*\S+"), "aws_secret"),
        (re.compile(r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private_key"),
        (re.compile(r"(?i)\bpassword\s*[:=]\s*[^\s,;]{6,}"), "password"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn_like"),
        (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "card_like"),
    ]
    UNSAFE_PATTERNS = [
        (re.compile(r"(?i)\brm\s+-rf\s+/(?:\s|$)"), "destructive_command"),
        (re.compile(r"(?i)\bmkfs(\.[a-z0-9]+)?\b"), "destructive_command"),
        (re.compile(r"(?i)\bdd\s+if=/dev/(zero|random)\s+of=/dev/"), "destructive_command"),
        (re.compile(r"(?i):\(\)\s*\{\s*:\|\s*:\s*&\s*\}\s*;\s*:"), "fork_bomb"),
    ]

    @classmethod
    def classify(cls, text: str) -> Dict[str, Any]:
        findings: List[Dict[str, str]] = []
        sample = text or ""
        for pattern, label in cls.SECRET_PATTERNS:
            if pattern.search(sample):
                findings.append({"type": "secret", "label": label})
        for pattern, label in cls.UNSAFE_PATTERNS:
            if pattern.search(sample):
                findings.append({"type": "unsafe", "label": label})
        severity = "clean"
        if any(f["type"] == "unsafe" for f in findings):
            severity = "block"
        elif findings:
            severity = "redact"
        return {"severity": severity, "findings": findings}

    @classmethod
    def sanitize(cls, text: str) -> Tuple[str, Dict[str, Any]]:
        verdict = cls.classify(text)
        if verdict["severity"] == "clean":
            return text, verdict
        out = text or ""
        for pattern, label in cls.SECRET_PATTERNS:
            out = pattern.sub(f"[REDACTED:{label}]", out)
        if verdict["severity"] == "block":
            for pattern, label in cls.UNSAFE_PATTERNS:
                out = pattern.sub(f"[BLOCKED:{label}]", out)
        return out, verdict


CLASSIFIER = OutputClassifier()


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, ctx: Dict[str, Any], spec: ProceduralSpec, tool: str, arguments: Dict[str, Any]) -> Tuple[Observation, Dict[str, Any]]:
        started = time.time()
        tenant_id = ctx["tenant_id"]
        run_id = ctx["run_id"]
        step = int(ctx.get("step", 0))
        try:
            receipt = ZeroTrustGate.verify(tenant_id, run_id, spec, tool, arguments)
        except (ToolDenied, SecurityError, ValidationError) as exc:
            audit(tenant_id, run_id, "runtime", f"tool_denied:{tool}", {"error": str(exc)}, False)
            obs = Observation(
                step=step,
                source="gate",
                tool=tool,
                ok=False,
                summary=f"authorization denied for {tool}",
                data={},
                error=str(exc),
                latency_ms=int((time.time() - started) * 1000),
                created_at=iso(),
            )
            return obs, {"denied": True, "error": str(exc)}
        try:
            handler = self.registry.get(tool).handler
            result = handler(ctx, **(arguments or {}))
            if not isinstance(result, dict):
                result = {"value": result}
            ok = bool(result.get("ok", True)) and not result.get("error")
            raw_summary = self._summarize(tool, result)
            summary, verdict = CLASSIFIER.sanitize(raw_summary)
            if verdict["severity"] == "block":
                ok = False
            obs = Observation(
                step=step,
                source="tool",
                tool=tool,
                ok=ok,
                summary=summary,
                data=self._prune(result),
                error=(str(result.get("error"))[:2000] if result.get("error") else (None if ok else "tool reported failure")),
                latency_ms=int((time.time() - started) * 1000),
                created_at=iso(),
            )
            receipt["classifier"] = verdict
            return obs, receipt
        except (SecurityError, ValidationError, ToolDenied) as exc:
            obs = Observation(
                step=step,
                source="tool",
                tool=tool,
                ok=False,
                summary=f"{tool} rejected: {exc}",
                data={},
                error=str(exc)[:2000],
                latency_ms=int((time.time() - started) * 1000),
                created_at=iso(),
            )
            return obs, receipt
        except FileNotFoundError as exc:
            obs = Observation(
                step=step,
                source="tool",
                tool=tool,
                ok=False,
                summary=f"{tool} target missing: {exc}",
                data={},
                error=str(exc)[:2000],
                latency_ms=int((time.time() - started) * 1000),
                created_at=iso(),
            )
            return obs, receipt
        except Exception as exc:
            log.warning("tool %s crashed: %s", tool, exc)
            obs = Observation(
                step=step,
                source="tool",
                tool=tool,
                ok=False,
                summary=f"{tool} raised {type(exc).__name__}",
                data={"traceback": traceback.format_exc()[-2000:]},
                error=f"{type(exc).__name__}: {exc}"[:2000],
                latency_ms=int((time.time() - started) * 1000),
                created_at=iso(),
            )
            return obs, receipt

    @staticmethod
    def _prune(result: Dict[str, Any], cap: int = 12000) -> Dict[str, Any]:
        encoded = jdump(result)
        if len(encoded) <= cap:
            return json.loads(encoded)
        pruned: Dict[str, Any] = {}
        for key, value in result.items():
            if isinstance(value, str) and len(value) > 2500:
                pruned[key] = value[:2500] + f"...[truncated {len(value) - 2500} chars]"
            elif isinstance(value, list) and len(value) > 40:
                pruned[key] = value[:40] + [f"...[truncated {len(value) - 40} items]"]
            elif isinstance(value, dict) and len(jdump(value)) > 3000:
                keys = list(value.keys())[:25]
                pruned[key] = {k: value[k] for k in keys}
            else:
                pruned[key] = value
        pruned["_truncated"] = True
        return json.loads(jdump(pruned))

    @staticmethod
    def _summarize(tool: str, result: Dict[str, Any]) -> str:
        if result.get("error"):
            return f"{tool} failed: {str(result['error'])[:600]}"
        if tool == "workspace.read_file":
            return f"read {result.get('path')} lines {result.get('start')}-{result.get('end')} of {result.get('total_lines')}:\n{str(result.get('content', ''))[:2500]}"
        if tool == "workspace.write_file":
            return f"wrote {result.get('path')} ({result.get('bytes')} bytes, {result.get('lines')} lines)"
        if tool == "workspace.append_file":
            return f"appended {result.get('added')} new lines to {result.get('path')} (skipped {result.get('skipped')} duplicates, total {result.get('total_lines')})"
        if tool == "workspace.replace_lines":
            return f"replaced lines {result.get('start')}-{result.get('end')} in {result.get('path')} with {result.get('inserted_lines')} lines (total {result.get('total_lines')})"
        if tool == "workspace.check_lines":
            return f"membership check on {result.get('path')}: found={len(result.get('found', []))} missing={len(result.get('missing', []))} missing_sample={result.get('missing', [])[:6]}"
        if tool == "workspace.list_dir":
            entries = result.get("entries", [])
            return f"{len(entries)} entries: " + ", ".join(f"{e.get('name')}({e.get('type')})" for e in entries[:25])
        if tool == "compute.python":
            return f"python ok={result.get('ok')} stdout={str(result.get('stdout', ''))[:1500]} result={str(result.get('result'))[:800]} err={str(result.get('error') or '')[:400]}"
        if tool == "compute.shell":
            return f"shell exit={result.get('exit_code')} stdout={str(result.get('stdout', ''))[:1500]} stderr={str(result.get('stderr', ''))[:600]}"
        if tool == "compute.http_get":
            return f"http {result.get('status')} {result.get('url')} :: {str(result.get('text', ''))[:2000]}"
        if tool == "memory.wiki_search":
            return "wiki hits: " + " | ".join(f"{r['title']}({r['score']:.3f}): {r['excerpt'][:200]}" for r in result.get("results", [])[:4])
        if tool == "memory.skill_search":
            return "skill hits: " + " | ".join(f"{r['skill']['name']}({r['score']:.3f})" for r in result.get("results", [])[:4])
        if tool == "memory.trace_query":
            traces = result.get("traces", [])
            return f"{len(traces)} traces: " + " | ".join(f"s{t.get('step')}:{t.get('kind')}:{'ok' if t.get('success') else 'fail'}" for t in traces[:12])
        if tool == "control.finish":
            return f"run finished: {str(result.get('summary'))[:2000]}"
        if tool == "control.fail":
            return f"run failed: {str(result.get('reason'))[:1500]}"
        return f"{tool} completed: {jdump(result)[:1800]}"


EXECUTOR = ToolExecutor(TOOLS)


STEP_SYSTEM_PROMPT = """You are the deterministic reasoning core of a long-horizon autonomous agent runtime.

CRITICAL CONTRACT
You never receive conversational history. Your entire input is exactly three structures:
  P  = immutable procedural specification (objective, success criteria, constraints)
  S  = structured execution state (the sufficient statistic of all prior progress)
  O  = the single latest environment observation
Your within-step reasoning is destroyed after this step. Anything that must survive to the next
step MUST be written into the state patch. Nothing else persists.

OUTPUT CONTRACT
Reply with exactly ONE JSON object and nothing else. No prose, no markdown fences, no commentary.

{
  "reasoning": "concise within-step analysis, at most 8 sentences",
  "state_patch": {
    "phase": "string",
    "current_subgoal": "string",
    "plan": ["ordered remaining steps"],
    "progress": ["verified completed facts to append"],
    "open_goals": ["unresolved goals to append"],
    "dependencies": ["blocking dependencies"],
    "constraints": ["hard constraints discovered"],
    "facts": {"key": "verified value"},
    "artifacts": {"logical_name": "workspace/path"},
    "metrics": {"name": 0},
    "errors": ["error signatures to append"],
    "skill_notes": ["reusable procedural insights"],
    "verification": {"criterion": true},
    "cursor": {"position": "resume marker"},
    "scratch": {"ephemeral": "value"}
  },
  "action": {
    "tool": "exact.tool.name",
    "arguments": {"...": "..."},
    "rationale": "one sentence justification"
  }
}

STATE PATCH SEMANTICS
- Include ONLY keys you are changing. Omit everything else.
- List keys (progress, open_goals, dependencies, constraints, errors, plan, skill_notes) APPEND with deduplication.
- Object keys (facts, artifacts, metrics, verification, cursor, scratch) MERGE key-by-key.
- Setting any key to null CLEARS that key (deletion primitive).
- To remove one entry inside an object key, set that inner key to null.

OPERATING RULES
1. Exactly one tool call per step. Never batch multiple actions.
2. Write durable knowledge into state_patch before acting; never rely on memory of this reasoning.
3. Use cursor to record exact resume markers for long file or dataset traversals.
4. When an observation reports failure, record the error signature and pivot strategy rather than retrying identically.
5. When every success criterion is verified in state.verification, call control.finish with a complete summary.
6. Call control.fail only when the objective is provably unachievable under the constraints.
7. Prefer memory.skill_upsert or memory.wiki_write to persist generalizable procedures you discover.
8. Keep the state compact: it is a sufficient statistic, not a transcript."""


class PromptBuilder:
    def __init__(self, registry: ToolRegistry, em: ExperientialMemory, wm: WorkingMemory):
        self.registry = registry
        self.em = em
        self.wm = wm

    def render_spec(self, spec: ProceduralSpec) -> str:
        lines = [f"OBJECTIVE: {spec.objective}"]
        if spec.success_criteria:
            lines.append("SUCCESS CRITERIA:")
            for i, c in enumerate(spec.success_criteria, 1):
                lines.append(f"  {i}. {c}")
        if spec.constraints:
            lines.append("HARD CONSTRAINTS:")
            for c in spec.constraints:
                lines.append(f"  - {c}")
        lines.append(f"STEP BUDGET: {spec.max_steps}")
        if spec.verifiers:
            lines.append("AUTOMATED VERIFIERS:")
            for v in spec.verifiers[:12]:
                lines.append(f"  - {v.get('type')}: {jdump({k: v[k] for k in v if k != 'type'})[:300]}")
        return "\n".join(lines)

    def render_sigma(self, sigma: Dict[str, Any]) -> str:
        view = {k: sigma.get(k) for k in SIGMA_ALLOWED_KEYS if sigma.get(k) not in (None, [], {}, "")}
        text = json.dumps(view, ensure_ascii=False, indent=1, sort_keys=True, default=str)
        if len(text) > 24000:
            text = text[:24000] + "\n...[state view truncated]"
        return text

    def render_observation(self, obs: Observation) -> str:
        lines = [
            f"SOURCE: {obs.source}",
            f"TOOL: {obs.tool or 'none'}",
            f"STATUS: {'ok' if obs.ok else 'FAILED'}",
            f"LATENCY_MS: {obs.latency_ms}",
        ]
        if obs.error:
            lines.append(f"ERROR: {obs.error[:1500]}")
        summary = obs.summary or ""
        lines.append("SUMMARY:")
        lines.append(summary[:6000] if summary else "(none)")
        data_text = jdump(obs.data)
        if data_text and data_text != "{}":
            lines.append("DATA:")
            lines.append(data_text[:6000])
        return "\n".join(lines)

    def build(
        self,
        spec: ProceduralSpec,
        sigma: Dict[str, Any],
        observation: Observation,
        step: int,
        skills: List[Tuple[Skill, float]],
        cognition: Optional[Dict[str, Any]] = None,
        reflection: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        blocks: List[str] = []
        blocks.append("=== P :: PROCEDURAL SPECIFICATION (IMMUTABLE) ===")
        blocks.append(self.render_spec(spec))
        blocks.append("")
        blocks.append("=== TOOL SURFACE ===")
        blocks.append(self.registry.describe(spec.allowed_tools))
        blocks.append("")
        if skills:
            blocks.append("=== EM :: RETRIEVED PROCEDURAL SKILLS ===")
            for skill, score in skills[:2]:
                blocks.append(f"[relevance={score:.4f}]")
                blocks.append(skill.render())
                blocks.append("")
        policy_hints = STORE.list_policy_hints(spec.tenant_id, 24)
        if policy_hints:
            blocks.append("=== LEARNED POLICY SIGNALS ===")
            for item in policy_hints:
                blocks.append(f"{item['feature']} weight={float(item['weight']):.4f} updates={int(item['updates'])}")
            blocks.append("")
        if cognition:
            blocks.append("=== SYSTEM-2 COGNITION DIRECTIVE ===")
            blocks.append(f"subgoal: {cognition.get('subgoal', '')}")
            blocks.append(f"gate: {float(cognition.get('gate', 1.0)):.4f}")
            blocks.append(f"staleness_ms: {int(cognition.get('staleness_ms', 0))}")
            blocks.append(f"staleness_encoding: {jdump(cognition.get('staleness_encoding', []))[:400]}")
            blocks.append(f"cognition_signature: {jdump(cognition.get('signature', []))[:600]}")
            blocks.append("")
        if reflection:
            blocks.append("=== REFLECTION PATCH (PRIVILEGED HINDSIGHT) ===")
            blocks.append(reflection[:4000])
            blocks.append("")
        blocks.append(f"=== S :: EXECUTION STATE AT STEP {step} ===")
        blocks.append(self.render_sigma(sigma))
        blocks.append("")
        blocks.append(f"=== O :: LATEST OBSERVATION (STEP {step}) ===")
        blocks.append(self.render_observation(observation))
        blocks.append("")
        blocks.append("Emit exactly one JSON object conforming to the OUTPUT CONTRACT now.")
        user = "\n".join(blocks)
        return [{"role": "system", "content": STEP_SYSTEM_PROMPT}, {"role": "user", "content": user}]


PROMPTS = PromptBuilder(TOOLS, EM, WM)


class Verifier:
    @staticmethod
    def evaluate(spec: ProceduralSpec, tenant_id: str, run_id: str, sigma: Dict[str, Any], terminal: Dict[str, Any]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for v in spec.verifiers:
            vtype = str(v.get("type") or "").strip()
            try:
                if vtype == "file_exists":
                    path = str(v.get("path") or "")
                    target = WORKSPACE.resolve(tenant_id, run_id, path)
                    ok = target.exists() and target.is_file()
                    results.append({"type": vtype, "path": path, "passed": ok, "detail": "present" if ok else "missing"})
                elif vtype == "file_contains":
                    path = str(v.get("path") or "")
                    needle = str(v.get("value") or "")
                    target = WORKSPACE.resolve(tenant_id, run_id, path)
                    text = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
                    ok = needle in text
                    results.append({"type": vtype, "path": path, "passed": ok, "detail": f"needle {'found' if ok else 'absent'}"})
                elif vtype == "file_min_lines":
                    path = str(v.get("path") or "")
                    minimum = int(v.get("value") or 1)
                    target = WORKSPACE.resolve(tenant_id, run_id, path)
                    count = len(target.read_text(encoding="utf-8", errors="replace").splitlines()) if target.exists() else 0
                    ok = count >= minimum
                    results.append({"type": vtype, "path": path, "passed": ok, "detail": f"{count} lines >= {minimum}"})
                elif vtype == "state_key_truthy":
                    key = str(v.get("key") or "")
                    node: Any = sigma
                    for part in key.split("."):
                        if isinstance(node, dict):
                            node = node.get(part)
                        else:
                            node = None
                            break
                    ok = bool(node)
                    results.append({"type": vtype, "key": key, "passed": ok, "detail": f"value={str(node)[:200]}"})
                elif vtype == "state_key_equals":
                    key = str(v.get("key") or "")
                    expected = v.get("value")
                    node = sigma
                    for part in key.split("."):
                        if isinstance(node, dict):
                            node = node.get(part)
                        else:
                            node = None
                            break
                    ok = node == expected
                    results.append({"type": vtype, "key": key, "passed": ok, "detail": f"actual={str(node)[:200]}"})
                elif vtype == "all_criteria_verified":
                    verification = sigma.get("verification") or {}
                    if not isinstance(verification, dict) or not spec.success_criteria:
                        ok = bool(verification) and all(bool(x) for x in verification.values())
                    else:
                        ok = len(verification) >= len(spec.success_criteria) and all(bool(x) for x in verification.values())
                    results.append({"type": vtype, "passed": ok, "detail": jdump(verification)[:400]})
                elif vtype == "python_assert":
                    code = str(v.get("code") or "")
                    run = PY_SANDBOX.run(code, {"sigma": json.loads(jdump(sigma)), "terminal": json.loads(jdump(terminal))})
                    ok = bool(run.get("ok")) and bool(run.get("result"))
                    results.append({"type": vtype, "passed": ok, "detail": f"result={str(run.get('result'))[:200]} err={str(run.get('error') or '')[:200]}"})
                elif vtype == "no_errors":
                    errs = sigma.get("errors") or []
                    ok = len(errs) == 0
                    results.append({"type": vtype, "passed": ok, "detail": f"{len(errs)} recorded errors"})
                else:
                    results.append({"type": vtype or "unknown", "passed": False, "detail": "unsupported verifier type"})
            except Exception as exc:
                results.append({"type": vtype or "unknown", "passed": False, "detail": f"verifier error: {exc}"[:300]})
        terminal_ok = str(terminal.get("status") or "") == "completed"
        if not spec.verifiers:
            verification = sigma.get("verification") or {}
            criteria_ok = True
            if spec.success_criteria:
                criteria_ok = bool(verification) and all(bool(x) for x in verification.values()) and len(verification) >= min(1, len(spec.success_criteria))
            passed = terminal_ok and criteria_ok
            results.append({"type": "terminal_declaration", "passed": terminal_ok, "detail": str(terminal.get("status"))})
            results.append({"type": "self_verification", "passed": criteria_ok, "detail": jdump(verification)[:400]})
        else:
            passed = terminal_ok and all(r["passed"] for r in results)
        score = (sum(1 for r in results if r["passed"]) / len(results)) if results else (1.0 if passed else 0.0)
        return {"passed": bool(passed), "score": float(score), "checks": results, "evaluated_at": iso()}


class ReflectionEngine:
    SYSTEM = """You are a hindsight reflection compiler for an autonomous agent runtime.
You receive a completed trajectory summary, the verifier report, and the terminal state.
Produce a compact Reflection Patch that a future agent could read BEFORE acting to avoid the same failures.

Reply with exactly ONE JSON object, no prose, no markdown:
{
  "verdict": "success" | "failure" | "partial",
  "root_cause": "single-sentence causal diagnosis",
  "failure_points": [{"step": 0, "what": "what went wrong", "why": "causal mechanism"}],
  "pivot_actions": [{"instead_of": "the wrong action", "do": "the correct action", "when": "trigger condition"}],
  "durable_rules": ["imperative rules that generalize beyond this task"],
  "skill_candidates": [{"name": "skill_name", "summary": "what it does", "procedure": ["step 1", "step 2"], "tags": ["tag"], "category": "category"}],
  "wiki_note": {"title": "page title", "category": "category", "body": "markdown body of durable caveats"},
  "patch_text": "dense imperative guidance block, at most 900 characters, written for injection above a future prompt"
}"""

    def __init__(self, model: ModelClient, store: SQLiteStore):
        self.model = model
        self.store = store

    def _trajectory_digest(self, run_id: str, limit: int = 90) -> str:
        traces = TRACES.for_run(run_id, limit)
        lines: List[str] = []
        for t in traces:
            action = t.get("action") or {}
            outcome = t.get("outcome") or {}
            lines.append(
                f"step={t['step']} tool={action.get('tool')} ok={t['success']} "
                f"summary={str(outcome.get('summary') or '')[:220]} err={str(outcome.get('error') or '')[:160]}"
            )
        return "\n".join(lines[-limit:])

    def generate(self, tenant_id: str, run_id: str, spec: ProceduralSpec, sigma: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
        digest = self._trajectory_digest(run_id)
        prompt = [
            {"role": "system", "content": self.SYSTEM},
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "=== OBJECTIVE ===",
                        spec.objective,
                        "",
                        "=== SUCCESS CRITERIA ===",
                        jdump(spec.success_criteria),
                        "",
                        "=== VERIFIER REPORT ===",
                        jdump(report)[:6000],
                        "",
                        "=== TERMINAL STATE ===",
                        jdump({k: sigma.get(k) for k in ("phase", "progress", "open_goals", "errors", "verification", "metrics", "artifacts")})[:8000],
                        "",
                        "=== TRAJECTORY DIGEST ===",
                        digest[:14000],
                        "",
                        "Emit the Reflection Patch JSON now.",
                    ]
                ),
            },
        ]
        parsed: Optional[Dict[str, Any]] = None
        if self.model.available:
            try:
                out = self.model.complete(prompt, override={"temperature": 0.3, "max_tokens": 6000, "frequency_penalty": 0.1, "presence_penalty": 0.0})
                parsed = DECODER.extract_json_object(out["text"])
                TENANTS.charge_tokens(tenant_id, out["usage"]["total_tokens"])
            except Exception as exc:
                log.warning("reflection generation failed: %s", exc)
        if not isinstance(parsed, dict):
            parsed = self._heuristic(spec, sigma, report, digest)
        patch = self._normalize(parsed, report)
        self.store.execute(
            "INSERT INTO reflection_patches(reflection_id, run_id, tenant_id, verdict, failure_points, pivot_actions, patch_text, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (new_id("refl"), run_id, tenant_id, patch["verdict"], jdump(patch["failure_points"]), jdump(patch["pivot_actions"]), patch["patch_text"], iso()),
        )
        self._consolidate(tenant_id, patch)
        return patch

    def _heuristic(self, spec: ProceduralSpec, sigma: Dict[str, Any], report: Dict[str, Any], digest: str) -> Dict[str, Any]:
        failed_checks = [c for c in report.get("checks", []) if not c.get("passed")]
        errors = [str(e) for e in (sigma.get("errors") or [])][-8:]
        failure_points: List[Dict[str, Any]] = []
        for line in digest.splitlines():
            if "ok=False" in line:
                m = re.search(r"step=(\d+)", line)
                failure_points.append({"step": int(m.group(1)) if m else 0, "what": line[:220], "why": "tool level failure recorded in trace"})
        failure_points = failure_points[-8:]
        verdict = "success" if report.get("passed") else ("partial" if float(report.get("score", 0.0)) > 0.4 else "failure")
        rules = [f"Verify criterion before finishing: {c}" for c in spec.success_criteria[:4]]
        for chk in failed_checks[:4]:
            rules.append(f"Ensure verifier '{chk.get('type')}' passes: {str(chk.get('detail'))[:160]}")
        for err in errors[:4]:
            rules.append(f"Avoid recurrence of error signature: {err[:160]}")
        patch_text = " ".join(
            ["HINDSIGHT:"]
            + [f"verdict={verdict}"]
            + [f"failed={len(failed_checks)}"]
            + rules[:6]
        )[:900]
        return {
            "verdict": verdict,
            "root_cause": (failed_checks[0].get("detail") if failed_checks else (errors[-1] if errors else "objective satisfied")),
            "failure_points": failure_points,
            "pivot_actions": [{"instead_of": "repeating the failing action", "do": "record the error signature in state and select an alternative tool path", "when": "an observation reports STATUS FAILED"}],
            "durable_rules": rules[:8],
            "skill_candidates": [],
            "wiki_note": {},
            "patch_text": patch_text,
        }

    @staticmethod
    def _normalize(parsed: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
        verdict = str(parsed.get("verdict") or ("success" if report.get("passed") else "failure")).lower()
        if verdict not in ("success", "failure", "partial"):
            verdict = "success" if report.get("passed") else "failure"
        fps = parsed.get("failure_points")
        failure_points: List[Dict[str, Any]] = []
        if isinstance(fps, list):
            for item in fps[:12]:
                if isinstance(item, dict):
                    failure_points.append({"step": int(item.get("step") or 0), "what": str(item.get("what") or "")[:400], "why": str(item.get("why") or "")[:400]})
                else:
                    failure_points.append({"step": 0, "what": str(item)[:400], "why": ""})
        pvs = parsed.get("pivot_actions")
        pivots: List[Dict[str, Any]] = []
        if isinstance(pvs, list):
            for item in pvs[:12]:
                if isinstance(item, dict):
                    pivots.append({"instead_of": str(item.get("instead_of") or "")[:300], "do": str(item.get("do") or "")[:300], "when": str(item.get("when") or "")[:200]})
                else:
                    pivots.append({"instead_of": "", "do": str(item)[:300], "when": ""})
        rules = [str(r)[:300] for r in (parsed.get("durable_rules") or []) if r][:12]
        skills: List[Dict[str, Any]] = []
        for cand in (parsed.get("skill_candidates") or [])[:6]:
            if not isinstance(cand, dict):
                continue
            name = str(cand.get("name") or "").strip()[:120]
            summary = str(cand.get("summary") or "").strip()[:600]
            proc = [str(p)[:400] for p in (cand.get("procedure") or []) if p][:20]
            if not name or not summary or not proc:
                continue
            skills.append(
                {
                    "name": name,
                    "summary": summary,
                    "procedure": proc,
                    "tags": [str(t)[:40] for t in (cand.get("tags") or [])][:10],
                    "category": str(cand.get("category") or "general")[:60],
                    "preconditions": [str(p)[:300] for p in (cand.get("preconditions") or [])][:10],
                    "failure_modes": [str(p)[:300] for p in (cand.get("failure_modes") or [])][:10],
                }
            )
        wiki = parsed.get("wiki_note") if isinstance(parsed.get("wiki_note"), dict) else {}
        patch_text = str(parsed.get("patch_text") or "").strip()
        if not patch_text:
            patch_text = " ".join(["HINDSIGHT:", f"verdict={verdict}"] + rules[:6])
        return {
            "verdict": verdict,
            "root_cause": str(parsed.get("root_cause") or "")[:600],
            "failure_points": failure_points,
            "pivot_actions": pivots,
            "durable_rules": rules,
            "skill_candidates": skills,
            "wiki_note": {
                "title": str(wiki.get("title") or "")[:160],
                "category": str(wiki.get("category") or "general")[:60],
                "body": str(wiki.get("body") or "")[:20000],
            },
            "patch_text": patch_text[:900],
        }

    def _consolidate(self, tenant_id: str, patch: Dict[str, Any]) -> None:
        for cand in patch.get("skill_candidates", []):
            try:
                EM.upsert(
                    tenant_id,
                    cand["name"],
                    cand["summary"],
                    cand["procedure"],
                    cand.get("preconditions"),
                    cand.get("failure_modes"),
                    cand.get("tags"),
                    cand.get("category", "general"),
                )
            except Exception as exc:
                log.warning("skill consolidation failed: %s", exc)
        note = patch.get("wiki_note") or {}
        if note.get("title") and note.get("body"):
            try:
                WIKI.upsert(tenant_id, note["title"], note["body"], note.get("category", "general"))
            except Exception as exc:
                log.warning("wiki consolidation failed: %s", exc)
        rules = patch.get("durable_rules") or []
        if rules:
            try:
                existing = WIKI.get_page(tenant_id, WIKI.slugify("Durable Operating Rules"))
                body_lines = (existing["body"].splitlines() if existing else [])
                have = set(l.strip("- ").strip() for l in body_lines)
                for r in rules:
                    if r.strip() and r.strip() not in have:
                        body_lines.append(f"- {r.strip()}")
                        have.add(r.strip())
                WIKI.upsert(tenant_id, "Durable Operating Rules", "\n".join(body_lines[-400:]), "playbook")
            except Exception as exc:
                log.warning("rule consolidation failed: %s", exc)


REFLECTION = ReflectionEngine(MODEL, STORE)


class TokenPolicyDistiller:
    def __init__(self, store: SQLiteStore, model: ModelClient):
        self.store = store
        self.model = model
        self._lock = threading.RLock()
        self.root = DISTILL_ROOT

    @staticmethod
    def token_split(text: str) -> List[str]:
        return re.findall(r"\s+|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text or "")

    def _features(self, prefix_tokens: List[str], token: str) -> List[str]:
        prev1 = prefix_tokens[-1].strip() if prefix_tokens else "<bos>"
        prev2 = prefix_tokens[-2].strip() if len(prefix_tokens) > 1 else "<bos>"
        tok = token.strip()
        feats = [
            f"uni::{tok[:32]}",
            f"bi::{prev1[:16]}|{tok[:16]}",
            f"tri::{prev2[:12]}|{prev1[:12]}|{tok[:12]}",
            f"cls::{'ws' if not tok else ('num' if tok.isdigit() else ('word' if tok.isalnum() else 'sym'))}",
            f"pos::{min(len(prefix_tokens) // 24, 24)}",
            f"len::{min(len(tok), 12)}",
        ]
        return feats

    def _weights(self, tenant_id: str, features: Iterable[str]) -> Dict[str, float]:
        feats = list(dict.fromkeys(features))
        if not feats:
            return {}
        placeholders = ",".join("?" for _ in feats)
        rows = self.store.query(
            f"SELECT feature, weight FROM policy_weights WHERE tenant_id = ? AND feature IN ({placeholders})",
            [tenant_id, *feats],
        )
        return {r["feature"]: float(r["weight"]) for r in rows}

    def _logprob(self, tenant_id: str, prefix: List[str], token: str, bias: float, cache: Dict[str, float]) -> float:
        feats = self._features(prefix, token)
        missing = [f for f in feats if f not in cache]
        if missing:
            cache.update(self._weights(tenant_id, missing))
            for f in missing:
                cache.setdefault(f, 0.0)
        score = bias + sum(cache.get(f, 0.0) for f in feats)
        return -math.log1p(math.exp(-clamp(score, -18.0, 18.0)))

    def score_pair(self, tenant_id: str, student_prompt: str, teacher_prompt: str, action_text: str) -> Dict[str, Any]:
        tokens = self.token_split(action_text)[:6000]
        if not tokens:
            return {"tokens": [], "student": [], "teacher": [], "reverse_kl": 0.0}
        cache: Dict[str, float] = {}
        student_bias = -0.55 - 0.10 * math.tanh(len(student_prompt) / 60000.0)
        teacher_bias = -0.18 - 0.05 * math.tanh(len(teacher_prompt) / 60000.0)
        student_lp: List[float] = []
        teacher_lp: List[float] = []
        prefix: List[str] = []
        for tok in tokens:
            student_lp.append(self._logprob(tenant_id, prefix, tok, student_bias, cache))
            teacher_lp.append(self._logprob(tenant_id, prefix, tok, teacher_bias, cache))
            prefix.append(tok)
            if len(prefix) > 512:
                prefix = prefix[-512:]
        rkl = 0.0
        for s, t in zip(student_lp, teacher_lp):
            pt = math.exp(clamp(t, -30.0, 0.0))
            rkl += pt * (t - s)
        rkl = rkl / max(1, len(tokens))
        return {"tokens": tokens, "student": student_lp, "teacher": teacher_lp, "reverse_kl": float(rkl)}

    def record(
        self,
        tenant_id: str,
        run_id: str,
        step: int,
        student_prompt: str,
        teacher_prompt: str,
        action_text: str,
        advantage: float,
    ) -> Dict[str, Any]:
        scored = self.score_pair(tenant_id, student_prompt, teacher_prompt, action_text)
        sample_id = new_id("dst")
        self.store.execute(
            "INSERT INTO distill_samples(sample_id, run_id, tenant_id, step, student_prompt, teacher_prompt, action_text, student_logprobs, teacher_logprobs, tokens, reverse_kl, advantage, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sample_id,
                run_id,
                tenant_id,
                int(step),
                student_prompt[-24000:],
                teacher_prompt[-24000:],
                action_text[:24000],
                jdump([round(x, 6) for x in scored["student"]]),
                jdump([round(x, 6) for x in scored["teacher"]]),
                jdump(scored["tokens"]),
                float(scored["reverse_kl"]),
                float(advantage),
            iso(),
            ),
        )
        return {"sample_id": sample_id, "reverse_kl": scored["reverse_kl"], "tokens": len(scored["tokens"])}

    def optimize(self, tenant_id: str, run_id: Optional[str] = None, lr: float = 0.04, epochs: int = 2, limit: int = 400) -> Dict[str, Any]:
        with self._lock:
            if run_id:
                rows = self.store.query(
                    "SELECT * FROM distill_samples WHERE tenant_id = ? AND run_id = ? ORDER BY step ASC LIMIT ?",
                    (tenant_id, run_id, int(limit)),
                )
            else:
                rows = self.store.query(
                    "SELECT * FROM distill_samples WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
                    (tenant_id, int(limit)),
                )
            if not rows:
                return {"updated_features": 0, "samples": 0, "mean_reverse_kl_before": 0.0, "mean_reverse_kl_after": 0.0, "epochs": 0}
            samples: List[Tuple[List[str], float]] = []
            before_vals: List[float] = []
            for r in rows:
                tokens = jload(r["tokens"], []) or []
                advantage = float(r["advantage"])
                before_vals.append(float(r["reverse_kl"]))
                samples.append((tokens, advantage))
            grads: Dict[str, float] = defaultdict(float)
            counts: Dict[str, int] = defaultdict(int)
            total_tokens = 0
            for epoch in range(max(1, int(epochs))):
                decay = 1.0 / (1.0 + epoch)
                for tokens, advantage in samples:
                    prefix: List[str] = []
                    for tok in tokens:
                        feats = self._features(prefix, tok)
                        for f in feats:
                            grads[f] += lr * decay * advantage / max(1.0, math.sqrt(len(feats)))
                            counts[f] += 1
                        prefix.append(tok)
                        if len(prefix) > 512:
                            prefix = prefix[-512:]
                        total_tokens += 1
            if not grads:
                return {"updated_features": 0, "samples": len(samples), "mean_reverse_kl_before": sum(before_vals) / len(before_vals), "mean_reverse_kl_after": sum(before_vals) / len(before_vals), "epochs": epochs}
            existing = self._weights(tenant_id, grads.keys())
            now = iso()
            payload: List[Tuple[Any, ...]] = []
            for feat, grad in grads.items():
                current = existing.get(feat, 0.0)
                updated = clamp(current * 0.995 + grad / max(1, counts[feat]) * 8.0, -6.0, 6.0)
                payload.append((new_id("pw"), tenant_id, feat, updated, counts[feat], now))
            self.store.executemany(
                "INSERT INTO policy_weights(weight_id, tenant_id, feature, weight, updates, updated_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(tenant_id, feature) DO UPDATE SET weight=excluded.weight, updates=policy_weights.updates+excluded.updates, updated_at=excluded.updated_at",
                payload,
            )
            after_vals: List[float] = []
            for r in rows:
                rescored = self.score_pair(tenant_id, r["student_prompt"], r["teacher_prompt"], r["action_text"])
                after_vals.append(float(rescored["reverse_kl"]))
                self.store.execute("UPDATE distill_samples SET reverse_kl = ? WHERE sample_id = ?", (float(rescored["reverse_kl"]), r["sample_id"]))
            report = {
                "updated_features": len(payload),
                "samples": len(samples),
                "tokens": total_tokens,
                "mean_reverse_kl_before": sum(before_vals) / max(1, len(before_vals)),
                "mean_reverse_kl_after": sum(after_vals) / max(1, len(after_vals)),
                "epochs": int(epochs),
                "learning_rate": lr,
                "optimized_at": iso(),
            }
            path = self.root / f"optimize_{re.sub(r'[^A-Za-z0-9_.-]', '_', tenant_id)}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(jdump(report) + "\n")
            return report


DISTILLER = TokenPolicyDistiller(STORE, MODEL)


class CognitionEngine:
    def __init__(self, store: SQLiteStore, k: int = COGNITION_K, h: int = COGNITION_H):
        self.store = store
        self.k = k
        self.h = h

    @staticmethod
    def sinusoidal_staleness(elapsed_ms: int, dim: int = 16) -> List[float]:
        out: List[float] = []
        t = max(0.0, float(elapsed_ms) / 1000.0)
        half = max(1, dim // 2)
        for i in range(half):
            freq = 1.0 / (10000.0 ** (2.0 * i / max(1, dim)))
            out.append(math.sin(t * freq))
            out.append(math.cos(t * freq))
        return [round(v, 6) for v in out[:dim]]

    def _project(self, sigma: Dict[str, Any], spec: ProceduralSpec) -> List[List[float]]:
        seeds = [
            spec.objective,
            str(sigma.get("phase") or ""),
            str(sigma.get("current_subgoal") or ""),
            jdump(sigma.get("plan") or [])[:2000],
            jdump(sigma.get("open_goals") or [])[:2000],
            jdump(sigma.get("errors") or [])[:2000],
            jdump(sigma.get("verification") or {})[:2000],
            jdump(sigma.get("facts") or {})[:2000],
        ]
        while len(seeds) < self.k:
            seeds.append("")
        tokens: List[List[float]] = []
        for i in range(self.k):
            base = EMBEDDER.embed(seeds[i] or f"slot_{i}")
            vec: List[float] = []
            stride = max(1, len(base) // self.h) if base else 1
            for j in range(self.h):
                idx = (j * stride) % max(1, len(base))
                vec.append(base[idx] if base else 0.0)
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            tokens.append([round(v, 6) for v in vec])
        return tokens

    def deliberate(self, tenant_id: str, run_id: str, step: int, spec: ProceduralSpec, sigma: Dict[str, Any]) -> Dict[str, Any]:
        tokens = self._project(sigma, spec)
        open_goals = list(sigma.get("open_goals") or [])
        errors = list(sigma.get("errors") or [])
        verification = sigma.get("verification") or {}
        criteria = max(1, len(spec.success_criteria) or 1)
        verified = sum(1 for v in verification.values() if v) if isinstance(verification, dict) else 0
        gate = clamp(0.25 + 0.55 * (verified / criteria) - 0.08 * min(6, len(errors)) + 0.05 * (1.0 if not open_goals else 0.0), 0.0, 1.0)
        subgoal = str(sigma.get("current_subgoal") or "").strip()
        if not subgoal:
            subgoal = str(open_goals[0]) if open_goals else (spec.success_criteria[0] if spec.success_criteria else spec.objective)
        flat: List[float] = []
        for row in tokens:
            flat.extend(row)
        cog_id = new_id("cog")
        created_ms = int(time.time() * 1000)
        self.store.execute(
            "INSERT INTO cognition_tokens(cog_id, run_id, tenant_id, step, vector, gate, subgoal, created_at, created_ms) VALUES(?,?,?,?,?,?,?,?,?)",
            (cog_id, run_id, tenant_id, int(step), pack_vector(flat), float(gate), subgoal[:600], iso(), created_ms),
        )
        self.store.execute(
            "DELETE FROM cognition_tokens WHERE run_id = ? AND cog_id NOT IN (SELECT cog_id FROM cognition_tokens WHERE run_id = ? ORDER BY step DESC, created_ms DESC LIMIT 24)",
            (run_id, run_id),
        )
        return {
            "cog_id": cog_id,
            "step": step,
            "gate": gate,
            "subgoal": subgoal,
            "shape": [self.k, self.h],
            "created_ms": created_ms,
            "signature": [round(sum(row) / max(1, len(row)), 6) for row in tokens],
        }

    def latest(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.store.one("SELECT * FROM cognition_tokens WHERE run_id = ? ORDER BY step DESC, created_ms DESC LIMIT 1", (run_id,))
        if row is None:
            return None
        vec = unpack_vector(row["vector"])
        rows_k = self.k if self.k > 0 else 1
        chunk = max(1, len(vec) // rows_k) if vec else 1
        signature = []
        for i in range(rows_k):
            seg = vec[i * chunk : (i + 1) * chunk]
            signature.append(round(sum(seg) / max(1, len(seg)), 6) if seg else 0.0)
        elapsed = max(0, int(time.time() * 1000) - int(row["created_ms"]))
        return {
            "cog_id": row["cog_id"],
            "step": int(row["step"]),
            "gate": float(row["gate"]),
            "subgoal": row["subgoal"],
            "staleness_ms": elapsed,
            "staleness_encoding": self.sinusoidal_staleness(elapsed),
            "signature": signature,
            "shape": [self.k, chunk],
        }


COGNITION = CognitionEngine(STORE)


class CheckpointManager:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def save(self, tenant_id: str, run_id: str, step: int, node: str, sigma: Dict[str, Any], observation: Observation, pending: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "run_id": run_id,
            "tenant_id": tenant_id,
            "step": step,
            "node": node,
            "sigma": sigma,
            "observation": observation.to_dict(),
            "pending": pending or {},
        }
        digest = stable_hash(payload)
        signature = sign_payload(payload)
        ckpt_id = new_id("ckpt")
        with self.store.tx() as c:
            c.execute(
                "INSERT INTO checkpoints(checkpoint_id, run_id, tenant_id, step, node, sigma, observation, pending, digest, signature, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(run_id, step, node) DO UPDATE SET sigma=excluded.sigma, observation=excluded.observation, pending=excluded.pending, digest=excluded.digest, signature=excluded.signature, created_at=excluded.created_at",
                (
                    ckpt_id,
                    run_id,
                    tenant_id,
                    int(step),
                    node,
                    jdump(sigma),
                    jdump(observation.to_dict()),
                    jdump(pending or {}),
                    digest,
                    signature,
                    iso(),
                ),
            )
            c.execute("UPDATE runs SET step = ?, updated_at = ? WHERE run_id = ?", (int(step), iso(), run_id))
        return ckpt_id

    def latest(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.store.one("SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step DESC, created_at DESC LIMIT 1", (run_id,))
        if row is None:
            return None
        sigma = jload(row["sigma"], empty_sigma()) or empty_sigma()
        obs_d = jload(row["observation"], {}) or {}
        payload = {
            "run_id": row["run_id"],
            "tenant_id": row["tenant_id"],
            "step": int(row["step"]),
            "node": row["node"],
            "sigma": sigma,
            "observation": obs_d,
            "pending": jload(row["pending"], {}) or {},
        }
        valid = verify_signature(payload, row["signature"])
        return {
            "checkpoint_id": row["checkpoint_id"],
            "step": int(row["step"]),
            "node": row["node"],
            "sigma": sigma,
            "observation": obs_d,
            "pending": payload["pending"],
            "integrity_ok": valid,
            "created_at": row["created_at"],
        }

    def prune(self, run_id: str, keep: int = 40) -> None:
        self.store.execute(
            "DELETE FROM checkpoints WHERE run_id = ? AND checkpoint_id NOT IN "
            "(SELECT checkpoint_id FROM checkpoints WHERE run_id = ? ORDER BY step DESC, created_at DESC LIMIT ?)",
            (run_id, run_id, int(keep)),
        )


CHECKPOINTS = CheckpointManager(STORE)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ERROR = "error"


class GraphNode(str, Enum):
    OBSERVE = "observe"
    DELIBERATE = "deliberate"
    ROUTE = "route"
    DECIDE = "decide"
    VALIDATE = "validate"
    ACT = "act"
    COMMIT = "commit"
    VERIFY = "verify"
    REFLECT = "reflect"
    TERMINATE = "terminate"


@dataclass
class RunHandle:
    run_id: str
    tenant_id: str
    conversation_id: Optional[str]
    spec: ProceduralSpec
    status: str
    step: int
    stop_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    started_at: float = field(default_factory=time.time)


class RunRepository:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def create(self, tenant_id: str, conversation_id: Optional[str], spec: ProceduralSpec) -> str:
        run_id = new_id("run")
        now = iso()
        self.store.execute(
            "INSERT INTO runs(run_id, tenant_id, conversation_id, spec, status, step, created_at, updated_at, tokens_used, wall_ms, resume_count) "
            "VALUES(?,?,?,?,?,0,?,?,0,0,0)",
            (run_id, tenant_id, conversation_id, jdump(spec.to_dict()), RunStatus.PENDING.value, now, now),
        )
        return run_id

    def get(self, run_id: str, tenant_id: Optional[str] = None) -> Optional[sqlite3.Row]:
        if tenant_id:
            return self.store.one("SELECT * FROM runs WHERE run_id = ? AND tenant_id = ?", (run_id, tenant_id))
        return self.store.one("SELECT * FROM runs WHERE run_id = ?", (run_id,))

    def list(self, tenant_id: str, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            rows = self.store.query(
                "SELECT run_id, tenant_id, conversation_id, status, step, created_at, updated_at, finished_at, verdict, tokens_used, wall_ms, error, spec, resume_count "
                "FROM runs WHERE tenant_id = ? AND status = ? ORDER BY created_at DESC LIMIT ?",
                (tenant_id, status, int(limit)),
            )
        else:
            rows = self.store.query(
                "SELECT run_id, tenant_id, conversation_id, status, step, created_at, updated_at, finished_at, verdict, tokens_used, wall_ms, error, spec, resume_count "
                "FROM runs WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
                (tenant_id, int(limit)),
            )
        out: List[Dict[str, Any]] = []
        for r in rows:
            spec = jload(r["spec"], {}) or {}
            out.append(
                {
                    "run_id": r["run_id"],
                    "tenant_id": r["tenant_id"],
                    "conversation_id": r["conversation_id"],
                    "status": r["status"],
                    "step": int(r["step"]),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "finished_at": r["finished_at"],
                    "verdict": jload(r["verdict"], None),
                    "tokens_used": int(r["tokens_used"]),
                    "wall_ms": int(r["wall_ms"]),
                    "error": r["error"],
                    "objective": spec.get("objective", ""),
                    "resume_count": int(r["resume_count"]),
                }
            )
        return out

    def set_status(self, run_id: str, status: str, error: Optional[str] = None) -> None:
        finished = iso() if status in (RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value, RunStatus.ERROR.value) else None
        self.store.execute(
            "UPDATE runs SET status = ?, updated_at = ?, finished_at = COALESCE(?, finished_at), error = COALESCE(?, error) WHERE run_id = ?",
            (status, iso(), finished, error, run_id),
        )

    def bump(self, run_id: str, step: int, tokens: int, wall_ms: int) -> None:
        self.store.execute(
            "UPDATE runs SET step = ?, tokens_used = tokens_used + ?, wall_ms = ?, updated_at = ? WHERE run_id = ?",
            (int(step), int(tokens), int(wall_ms), iso(), run_id),
        )

    def set_terminal(self, run_id: str, terminal_state: Dict[str, Any], verdict: Dict[str, Any]) -> None:
        self.store.execute(
            "UPDATE runs SET terminal_state = ?, verdict = ?, updated_at = ? WHERE run_id = ?",
            (jdump(terminal_state), jdump(verdict), iso(), run_id),
        )

    def acquire_lease(self, run_id: str, owner: str, ttl_s: int = 120) -> bool:
        now = utcnow()
        expires = (now + timedelta(seconds=ttl_s)).isoformat()
        with self.store.tx() as c:
            row = c.execute("SELECT lease_owner, lease_expires_at FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return False
            current_owner = row[0]
            current_exp = row[1]
            if current_owner and current_owner != owner and current_exp:
                try:
                    if datetime.fromisoformat(current_exp) > now:
                        return False
                except Exception:
                    pass
            c.execute("UPDATE runs SET lease_owner = ?, lease_expires_at = ? WHERE run_id = ?", (owner, expires, run_id))
            return True

    def renew_lease(self, run_id: str, owner: str, ttl_s: int = 120) -> None:
        expires = (utcnow() + timedelta(seconds=ttl_s)).isoformat()
        self.store.execute("UPDATE runs SET lease_owner = ?, lease_expires_at = ? WHERE run_id = ?", (owner, expires, run_id))

    def release_lease(self, run_id: str, owner: str) -> None:
        self.store.execute("UPDATE runs SET lease_owner = NULL, lease_expires_at = NULL WHERE run_id = ? AND lease_owner = ?", (run_id, owner))

    def increment_resume(self, run_id: str) -> None:
        self.store.execute("UPDATE runs SET resume_count = resume_count + 1, updated_at = ? WHERE run_id = ?", (iso(), run_id))

    def resumable(self, limit: int = 200) -> List[sqlite3.Row]:
        return self.store.query(
            "SELECT * FROM runs WHERE status IN (?, ?, ?) ORDER BY updated_at ASC LIMIT ?",
            (RunStatus.RUNNING.value, RunStatus.PENDING.value, RunStatus.PAUSED.value, int(limit)),
        )


RUNS = RunRepository(STORE)


class MessageRepository:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def ensure_conversation(self, tenant_id: str, conversation_id: Optional[str], title: str = "New conversation") -> str:
        if conversation_id:
            row = self.store.one("SELECT conversation_id FROM conversations WHERE conversation_id = ? AND tenant_id = ?", (conversation_id, tenant_id))
            if row is not None:
                return conversation_id
        cid = conversation_id or new_id("conv")
        now = iso()
        self.store.execute(
            "INSERT INTO conversations(conversation_id, tenant_id, title, created_at, updated_at, archived) VALUES(?,?,?,?,?,0) "
            "ON CONFLICT(conversation_id) DO UPDATE SET updated_at=excluded.updated_at",
            (cid, tenant_id, title[:200], now, now),
        )
        return cid

    def add(self, tenant_id: str, conversation_id: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        seq = self.store.next_seq(f"msg:{conversation_id}")
        mid = new_id("msg")
        now = iso()
        self.store.execute(
            "INSERT INTO messages(message_id, conversation_id, tenant_id, role, content, meta, created_at, seq) VALUES(?,?,?,?,?,?,?,?)",
            (mid, conversation_id, tenant_id, role, content, jdump(meta or {}), now, seq),
        )
        self.store.execute("UPDATE conversations SET updated_at = ? WHERE conversation_id = ?", (now, conversation_id))
        return {"message_id": mid, "conversation_id": conversation_id, "role": role, "content": content, "meta": meta or {}, "created_at": now, "seq": seq}

    def history(self, tenant_id: str, conversation_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        rows = self.store.query(
            "SELECT * FROM messages WHERE conversation_id = ? AND tenant_id = ? ORDER BY seq ASC LIMIT ?",
            (conversation_id, tenant_id, int(limit)),
        )
        return [
            {
                "message_id": r["message_id"],
                "role": r["role"],
                "content": r["content"],
                "meta": jload(r["meta"], {}),
                "created_at": r["created_at"],
                "seq": int(r["seq"]),
            }
            for r in rows
        ]

    def conversations(self, tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.store.query(
            "SELECT c.conversation_id, c.title, c.created_at, c.updated_at, "
            "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id) AS message_count "
            "FROM conversations c WHERE c.tenant_id = ? AND c.archived = 0 ORDER BY c.updated_at DESC LIMIT ?",
            (tenant_id, int(limit)),
        )
        return [dict(r) for r in rows]

    def rename(self, tenant_id: str, conversation_id: str, title: str) -> None:
        self.store.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ? AND tenant_id = ?",
            (title[:200], iso(), conversation_id, tenant_id),
        )

    def delete(self, tenant_id: str, conversation_id: str) -> None:
        self.store.execute("DELETE FROM conversations WHERE conversation_id = ? AND tenant_id = ?", (conversation_id, tenant_id))


MESSAGES = MessageRepository(STORE)


class StateTransitionKernel:
    def __init__(self):
        self.model = MODEL
        self.prompts = PROMPTS
        self.executor = EXECUTOR
        self.validator = VALIDATOR

    def decide(
        self,
        tenant_id: str,
        run_id: str,
        spec: ProceduralSpec,
        sigma: Dict[str, Any],
        observation: Observation,
        step: int,
        skills: List[Tuple[Skill, float]],
        cognition: Optional[Dict[str, Any]],
        reflection: Optional[str],
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        messages = self.prompts.build(spec, sigma, observation, step, skills, cognition, reflection)
        student_messages = self.prompts.build(spec, sigma, observation, step, skills, cognition, None)
        prompt_chars = sum(len(m["content"]) for m in messages)
        attempts = 0
        last_error: Optional[str] = None
        while attempts < 3:
            attempts += 1
            result = self.model.stream(messages, on_delta=on_delta)
            raw = result["text"]
            usage = result["usage"]
            parsed = DECODER.extract_json_object(raw)
            if parsed is None:
                last_error = "no parseable JSON object in model output"
                messages = messages + [
                    {"role": "assistant", "content": raw[-4000:]},
                    {"role": "user", "content": "Your previous output was not a single parseable JSON object. Emit ONLY the JSON object required by the OUTPUT CONTRACT."},
                ]
                continue
            try:
                patch = self.validator.validate_patch(parsed.get("state_patch"))
            except ValidationError as exc:
                last_error = f"state_patch rejected: {exc}"
                messages = messages + [
                    {"role": "assistant", "content": raw[-4000:]},
                    {"role": "user", "content": f"Your state_patch was rejected by the deterministic validator: {exc}. Re-emit a corrected single JSON object."},
                ]
                continue
            action = parsed.get("action")
            if not isinstance(action, dict) or not action.get("tool"):
                last_error = "action.tool missing"
                messages = messages + [
                    {"role": "assistant", "content": raw[-4000:]},
                    {"role": "user", "content": "Your output lacked action.tool. Re-emit the full JSON object with a valid action.tool from the TOOL SURFACE."},
                ]
                continue
            tool = str(action.get("tool")).strip()
            arguments = action.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            if tool not in TOOLS.names():
                candidates = [n for n in TOOLS.names() if n.endswith("." + tool) or n.split(".")[-1] == tool]
                if len(candidates) == 1:
                    tool = candidates[0]
                else:
                    last_error = f"unknown tool '{tool}'"
                    messages = messages + [
                        {"role": "assistant", "content": raw[-4000:]},
                        {"role": "user", "content": f"Tool '{tool}' does not exist. Choose exactly one tool from the TOOL SURFACE and re-emit the JSON object."},
                    ]
                    continue
            reasoning = str(parsed.get("reasoning") or "")[:8000]
            return {
                "ok": True,
                "attempts": attempts,
                "raw": raw,
                "reasoning": reasoning,
                "state_patch": patch,
                "tool": tool,
                "arguments": arguments,
                "rationale": str(action.get("rationale") or "")[:1000],
                "usage": usage,
                "prompt_chars": prompt_chars,
                "teacher_prompt": "\n".join(m["content"] for m in messages),
                "student_prompt": "\n".join(m["content"] for m in student_messages),
            }
        return {
            "ok": False,
            "attempts": attempts,
            "error": last_error or "decision failed",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "prompt_chars": prompt_chars,
            "teacher_prompt": "\n".join(m["content"] for m in messages),
            "student_prompt": "\n".join(m["content"] for m in student_messages),
        }


KERNEL = StateTransitionKernel()


class AgentOrchestrator:
    def __init__(self):
        self.handles: Dict[str, RunHandle] = {}
        self._lock = threading.RLock()
        self.executor = EXECUTOR
        self.owner = f"{os.environ.get('HOSTNAME', 'local')}:{os.getpid()}"
        self._shutdown = threading.Event()

    def _handle(self, run_id: str) -> Optional[RunHandle]:
        with self._lock:
            return self.handles.get(run_id)

    def active_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "run_id": h.run_id,
                    "tenant_id": h.tenant_id,
                    "status": h.status,
                    "step": h.step,
                    "paused": h.pause_event.is_set(),
                    "uptime_s": round(time.time() - h.started_at, 2),
                }
                for h in self.handles.values()
            ]

    def start(self, tenant_id: str, conversation_id: Optional[str], spec: ProceduralSpec, resume_run_id: Optional[str] = None) -> str:
        if resume_run_id:
            row = RUNS.get(resume_run_id, tenant_id)
            if row is None:
                raise ValidationError(f"run {resume_run_id} not found")
            run_id = resume_run_id
            RUNS.increment_resume(run_id)
            spec = ProceduralSpec.from_dict(jload(row["spec"], {}) or {})
            conversation_id = conversation_id or row["conversation_id"]
        else:
            run_id = RUNS.create(tenant_id, conversation_id, spec)
        if not RUNS.acquire_lease(run_id, self.owner):
            raise ValidationError(f"run {run_id} is leased by another worker")
        handle = RunHandle(
            run_id=run_id,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            spec=spec,
            status=RunStatus.RUNNING.value,
            step=int(RUNS.get(run_id)["step"] or 0),
        )
        with self._lock:
            existing = self.handles.get(run_id)
            if existing and existing.thread and existing.thread.is_alive():
                return run_id
            self.handles[run_id] = handle
        thread = threading.Thread(target=self._loop, args=(handle,), name=f"agent-{run_id}", daemon=True)
        handle.thread = thread
        RUNS.set_status(run_id, RunStatus.RUNNING.value)
        emit_event("run_started", run_id, tenant_id, {"objective": spec.objective, "spec": spec.to_dict(), "resumed": bool(resume_run_id)}, conversation_id)
        thread.start()
        return run_id

    def pause(self, run_id: str) -> bool:
        h = self._handle(run_id)
        if h is None:
            return False
        h.pause_event.set()
        h.status = RunStatus.PAUSED.value
        RUNS.set_status(run_id, RunStatus.PAUSED.value)
        emit_event("run_paused", run_id, h.tenant_id, {}, h.conversation_id)
        return True

    def resume_paused(self, run_id: str) -> bool:
        h = self._handle(run_id)
        if h is None:
            return False
        h.pause_event.clear()
        h.status = RunStatus.RUNNING.value
        RUNS.set_status(run_id, RunStatus.RUNNING.value)
        emit_event("run_resumed", run_id, h.tenant_id, {}, h.conversation_id)
        return True

    def cancel(self, run_id: str) -> bool:
        h = self._handle(run_id)
        if h is None:
            RUNS.set_status(run_id, RunStatus.CANCELLED.value)
            return True
        h.status = RunStatus.CANCELLED.value
        h.stop_event.set()
        h.pause_event.clear()
        return True

    def shutdown(self, timeout: float = 12.0) -> None:
        self._shutdown.set()
        with self._lock:
            handles = list(self.handles.values())
        for h in handles:
            h.stop_event.set()
        deadline = time.time() + timeout
        for h in handles:
            if h.thread is not None:
                remaining = max(0.1, deadline - time.time())
                h.thread.join(remaining)

    def recover(self) -> int:
        rows = RUNS.resumable()
        recovered = 0
        for row in rows:
            run_id = row["run_id"]
            tenant_id = row["tenant_id"]
            spec = ProceduralSpec.from_dict(jload(row["spec"], {}) or {})
            if self._handle(run_id) is not None:
                continue
            try:
                TENANTS.get(tenant_id)
            except SecurityError:
                RUNS.set_status(run_id, RunStatus.ERROR.value, "tenant unavailable during recovery")
                continue
            try:
                self.start(tenant_id, row["conversation_id"], spec, resume_run_id=run_id)
                recovered += 1
                log.info("recovered run %s at step %s", run_id, row["step"])
            except Exception as exc:
                log.warning("recovery failed for %s: %s", run_id, exc)
        return recovered

    def _load_state(self, handle: RunHandle) -> Tuple[Dict[str, Any], Observation, int]:
        ckpt = CHECKPOINTS.latest(handle.run_id)
        if ckpt is None:
            sigma = empty_sigma()
            sigma["phase"] = "bootstrap"
            sigma["open_goals"] = list(handle.spec.success_criteria) or [handle.spec.objective]
            sigma["constraints"] = list(handle.spec.constraints)
            sigma["plan"] = ["analyze objective", "decompose into verifiable subgoals", "execute", "verify each success criterion", "finish"]
            obs = Observation.initial(0, f"Run initialized. Objective: {handle.spec.objective[:1200]}")
            CHECKPOINTS.save(handle.tenant_id, handle.run_id, 0, GraphNode.OBSERVE.value, sigma, obs, {})
            return sigma, obs, 0
        if not ckpt.get("integrity_ok"):
            log.warning("checkpoint integrity mismatch for run %s at step %s", handle.run_id, ckpt["step"])
            emit_event("checkpoint_integrity_warning", handle.run_id, handle.tenant_id, {"step": ckpt["step"]}, handle.conversation_id)
        sigma = ckpt["sigma"] or empty_sigma()
        obs_d = ckpt["observation"] or {}
        obs = Observation(
            step=int(obs_d.get("step") or ckpt["step"]),
            source=str(obs_d.get("source") or "checkpoint"),
            tool=obs_d.get("tool"),
            ok=bool(obs_d.get("ok", True)),
            summary=str(obs_d.get("summary") or f"resumed from checkpoint at step {ckpt['step']}"),
            data=obs_d.get("data") or {},
            error=obs_d.get("error"),
            latency_ms=int(obs_d.get("latency_ms") or 0),
            created_at=str(obs_d.get("created_at") or iso()),
        )
        return sigma, obs, int(ckpt["step"])

    def _reflection_hint(self, tenant_id: str, spec: ProceduralSpec) -> Optional[str]:
        rows = STORE.query(
            "SELECT patch_text FROM reflection_patches WHERE tenant_id = ? AND verdict IN ('failure','partial') ORDER BY created_at DESC LIMIT 3",
            (tenant_id,),
        )
        if not rows:
            return None
        texts = [r["patch_text"] for r in rows if r["patch_text"]]
        if not texts:
            return None
        return "\n".join(texts)[:3500]

    def _loop(self, handle: RunHandle) -> None:
        run_id = handle.run_id
        tenant_id = handle.tenant_id
        spec = handle.spec
        started_wall = time.time()
        sigma, observation, step = self._load_state(handle)
        WM.sync_from_sigma(tenant_id, run_id, sigma)
        reflection_hint = self._reflection_hint(tenant_id, spec)
        terminal: Dict[str, Any] = {}
        last_deliberation = 0.0
        cognition_state: Optional[Dict[str, Any]] = COGNITION.latest(run_id)
        min_interval = 1.0 / max(0.5, SYSTEM1_HZ)
        deliberate_interval = 1.0 / max(0.05, SYSTEM2_HZ)
        consecutive_failures = 0
        try:
            while not handle.stop_event.is_set() and not self._shutdown.is_set():
                if handle.pause_event.is_set():
                    time.sleep(0.4)
                    RUNS.renew_lease(run_id, self.owner)
                    continue
                if step >= spec.max_steps:
                    terminal = {"status": "failed", "reason": f"step budget exhausted at {step} steps"}
                    break
                if time.time() - started_wall > DEFAULT_WALL_BUDGET_S:
                    terminal = {"status": "failed", "reason": "wall clock budget exhausted"}
                    break
                cycle_started = time.time()
                step += 1
                handle.step = step
                RUNS.renew_lease(run_id, self.owner)

                if (time.time() - last_deliberation) >= deliberate_interval or cognition_state is None:
                    try:
                        COGNITION.deliberate(tenant_id, run_id, step, spec, sigma)
                        cognition_state = COGNITION.latest(run_id)
                        last_deliberation = time.time()
                        emit_event(
                            "cognition",
                            run_id,
                            tenant_id,
                            {"step": step, "gate": cognition_state.get("gate") if cognition_state else None, "subgoal": cognition_state.get("subgoal") if cognition_state else ""},
                            handle.conversation_id,
                        )
                    except Exception as exc:
                        log.warning("deliberation failed: %s", exc)

                routing_query = WM.routing_query(spec, sigma, observation)
                try:
                    skills = EM.search(tenant_id, routing_query, 2)
                except Exception as exc:
                    log.warning("skill routing failed: %s", exc)
                    skills = []
                if skills:
                    emit_event("skills_routed", run_id, tenant_id, {"step": step, "skills": [{"name": s.name, "score": round(sc, 5)} for s, sc in skills]}, handle.conversation_id)

                CHECKPOINTS.save(tenant_id, run_id, step, GraphNode.DECIDE.value, sigma, observation, {"routing_query": routing_query[:2000]})

                emit_event("step_begin", run_id, tenant_id, {"step": step, "phase": sigma.get("phase"), "subgoal": sigma.get("current_subgoal")}, handle.conversation_id)

                stream_buffer: List[str] = []

                def on_delta(chunk: str) -> None:
                    stream_buffer.append(chunk)
                    if len(stream_buffer) % 6 == 0:
                        emit_event("model_delta", run_id, tenant_id, {"step": step, "delta": "".join(stream_buffer[-6:])}, handle.conversation_id)

                decision = KERNEL.decide(
                    tenant_id,
                    run_id,
                    spec,
                    sigma,
                    observation,
                    step,
                    skills,
                    cognition_state,
                    reflection_hint,
                    on_delta,
                )
                usage = decision.get("usage") or {"total_tokens": 0}
                try:
                    TENANTS.charge_tokens(tenant_id, int(usage.get("total_tokens") or 0))
                except BudgetExceeded as exc:
                    terminal = {"status": "failed", "reason": str(exc)}
                    break
                RUNS.bump(run_id, step, int(usage.get("total_tokens") or 0), int((time.time() - started_wall) * 1000))

                if not decision.get("ok"):
                    consecutive_failures += 1
                    err = str(decision.get("error") or "decision failure")
                    sigma = VALIDATOR.apply(sigma, {"errors": [f"step{step}:decode:{err[:200]}"]})
                    observation = Observation(
                        step=step,
                        source="runtime",
                        tool=None,
                        ok=False,
                        summary=f"decision layer failed at step {step}: {err[:400]}. Re-emit a valid JSON object.",
                        data={},
                        error=err[:1000],
                        latency_ms=int((time.time() - cycle_started) * 1000),
                        created_at=iso(),
                    )
                    TRACES.append(tenant_id, run_id, step, "decision_failure", {"phase": sigma.get("phase")}, {"tool": None}, {"error": err}, {}, False, observation.latency_ms)
                    CHECKPOINTS.save(tenant_id, run_id, step, GraphNode.VALIDATE.value, sigma, observation, {})
                    emit_event("step_error", run_id, tenant_id, {"step": step, "error": err[:800]}, handle.conversation_id)
                    if consecutive_failures >= 8:
                        terminal = {"status": "failed", "reason": f"decision layer failed {consecutive_failures} consecutive times: {err[:400]}"}
                        break
                    time.sleep(min(8.0, 0.6 * consecutive_failures))
                    continue

                pre_sigma = json.loads(jdump(sigma))
                patch = decision["state_patch"]
                try:
                    candidate_sigma = VALIDATOR.apply(sigma, patch)
                except Exception as exc:
                    log.warning("state merge failed, rolling back: %s", exc)
                    candidate_sigma = pre_sigma
                    patch = {}
                    emit_event("state_rollback", run_id, tenant_id, {"step": step, "error": str(exc)[:400]}, handle.conversation_id)

                tool = decision["tool"]
                arguments = decision["arguments"]
                emit_event(
                    "action_selected",
                    run_id,
                    tenant_id,
                    {"step": step, "tool": tool, "rationale": decision.get("rationale", ""), "arguments": ToolExecutor._prune(arguments, 3000)},
                    handle.conversation_id,
                )

                ctx = {"tenant_id": tenant_id, "run_id": run_id, "step": step, "conversation_id": handle.conversation_id}
                obs_next, receipt = self.executor.execute(ctx, spec, tool, arguments)

                skill_id = skills[0][0].skill_id if skills else None
                if skill_id:
                    try:
                        EM.record_outcome(skill_id, obs_next.ok)
                    except Exception as exc:
                        log.debug("skill outcome record failed: %s", exc)

                TRACES.append(
                    tenant_id,
                    run_id,
                    step,
                    "step",
                    {"phase": pre_sigma.get("phase"), "current_subgoal": pre_sigma.get("current_subgoal"), "open_goals": pre_sigma.get("open_goals", [])[:8]},
                    {"tool": tool, "arguments": ToolExecutor._prune(arguments, 4000), "rationale": decision.get("rationale", "")},
                    {"ok": obs_next.ok, "summary": obs_next.summary[:3000], "error": obs_next.error, "latency_ms": obs_next.latency_ms},
                    patch,
                    obs_next.ok,
                    obs_next.latency_ms,
                    skill_id,
                    receipt,
                )

                sigma = candidate_sigma
                if obs_next.ok:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    sigma = VALIDATOR.apply(sigma, {"errors": [f"step{step}:{tool}:{(obs_next.error or 'failed')[:180]}"]})
                    reflection_hint = self._reflection_hint(tenant_id, spec)

                WM.sync_from_sigma(tenant_id, run_id, sigma)
                CHECKPOINTS.save(tenant_id, run_id, step, GraphNode.COMMIT.value, sigma, obs_next, {})
                CHECKPOINTS.prune(run_id)

                try:
                    advantage = 1.0 if obs_next.ok else -1.0
                    DISTILLER.record(
                        tenant_id,
                        run_id,
                        step,
                        decision.get("student_prompt", "")[-20000:],
                        decision.get("teacher_prompt", "")[-20000:],
                        jdump({"reasoning": decision.get("reasoning", "")[:2000], "state_patch": patch, "action": {"tool": tool, "arguments": arguments}}),
                        advantage,
                    )
                except Exception as exc:
                    log.debug("distill record failed: %s", exc)

                emit_event(
                    "step_complete",
                    run_id,
                    tenant_id,
                    {
                        "step": step,
                        "tool": tool,
                        "ok": obs_next.ok,
                        "summary": obs_next.summary[:2500],
                        "error": obs_next.error,
                        "state": {
                            "phase": sigma.get("phase"),
                            "current_subgoal": sigma.get("current_subgoal"),
                            "progress": (sigma.get("progress") or [])[-6:],
                            "open_goals": (sigma.get("open_goals") or [])[:6],
                            "verification": sigma.get("verification") or {},
                            "errors": (sigma.get("errors") or [])[-4:],
                        },
                        "tokens": int(usage.get("total_tokens") or 0),
                        "prompt_chars": decision.get("prompt_chars", 0),
                        "reasoning": decision.get("reasoning", "")[:1500],
                    },
                    handle.conversation_id,
                )

                observation = obs_next
                if isinstance(obs_next.data, dict) and obs_next.data.get("terminal"):
                    terminal = {
                        "status": str(obs_next.data.get("status") or ("completed" if obs_next.ok else "failed")),
                        "summary": obs_next.data.get("summary") or obs_next.data.get("reason") or "",
                        "artifacts": obs_next.data.get("artifacts") or {},
                    }
                    break
                if consecutive_failures >= 12:
                    terminal = {"status": "failed", "reason": f"12 consecutive tool failures; last error: {(obs_next.error or '')[:400]}"}
                    break

                elapsed = time.time() - cycle_started
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

            if handle.stop_event.is_set() and not terminal:
                terminal = {"status": "cancelled", "reason": "cancelled by operator"}
            if self._shutdown.is_set() and not terminal:
                terminal = {"status": "paused", "reason": "worker shutdown; run is checkpointed and resumable"}
            if not terminal:
                terminal = {"status": "failed", "reason": "loop exited without terminal declaration"}

            self._finalize(handle, sigma, observation, step, terminal, started_wall)
        except BudgetExceeded as exc:
            self._finalize(handle, sigma, observation, step, {"status": "failed", "reason": str(exc)}, started_wall)
        except Exception as exc:
            log.exception("run %s crashed", run_id)
            RUNS.set_status(run_id, RunStatus.ERROR.value, f"{type(exc).__name__}: {exc}"[:2000])
            emit_event("run_error", run_id, tenant_id, {"error": f"{type(exc).__name__}: {exc}"[:1500], "step": step, "resumable": True}, handle.conversation_id)
        finally:
            RUNS.release_lease(run_id, self.owner)
            with self._lock:
                self.handles.pop(run_id, None)

    def _finalize(self, handle: RunHandle, sigma: Dict[str, Any], observation: Observation, step: int, terminal: Dict[str, Any], started_wall: float) -> None:
        run_id = handle.run_id
        tenant_id = handle.tenant_id
        spec = handle.spec
        status_raw = str(terminal.get("status") or "failed")
        CHECKPOINTS.save(tenant_id, run_id, step, GraphNode.TERMINATE.value, sigma, observation, {"terminal": terminal})
        if status_raw == "paused":
            RUNS.set_status(run_id, RunStatus.PAUSED.value)
            emit_event("run_paused", run_id, tenant_id, {"step": step, "reason": terminal.get("reason", "")}, handle.conversation_id)
            return
        if status_raw == "cancelled":
            RUNS.set_status(run_id, RunStatus.CANCELLED.value)
            emit_event("run_cancelled", run_id, tenant_id, {"step": step, "reason": terminal.get("reason", "")}, handle.conversation_id)
            return
        report = Verifier.evaluate(spec, tenant_id, run_id, sigma, terminal)
        RUNS.set_terminal(run_id, {"sigma": sigma, "terminal": terminal, "step": step}, report)
        final_status = RunStatus.COMPLETED.value if report["passed"] else RunStatus.FAILED.value
        RUNS.set_status(run_id, final_status, None if report["passed"] else str(terminal.get("reason") or "verification failed")[:2000])
        RUNS.bump(run_id, step, 0, int((time.time() - started_wall) * 1000))
        emit_event("verification", run_id, tenant_id, {"report": report, "step": step}, handle.conversation_id)

        final_text = self._compose_final(spec, sigma, terminal, report)
        if handle.conversation_id:
            MESSAGES.add(
                tenant_id,
                handle.conversation_id,
                "assistant",
                final_text,
                {"run_id": run_id, "steps": step, "passed": report["passed"], "score": report["score"]},
            )
        emit_event(
            "run_finished",
            run_id,
            tenant_id,
            {
                "status": final_status,
                "steps": step,
                "passed": report["passed"],
                "score": report["score"],
                "summary": final_text[:6000],
                "terminal": terminal,
            },
            handle.conversation_id,
        )
        try:
            patch = REFLECTION.generate(tenant_id, run_id, spec, sigma, report)
            emit_event("reflection", run_id, tenant_id, {"verdict": patch["verdict"], "root_cause": patch["root_cause"], "rules": patch["durable_rules"][:6]}, handle.conversation_id)
        except Exception as exc:
            log.warning("reflection stage failed: %s", exc)
        try:
            advantage_report = DISTILLER.optimize(tenant_id, run_id, lr=0.05 if report["passed"] else 0.03, epochs=2)
            emit_event("distillation", run_id, tenant_id, advantage_report, handle.conversation_id)
        except Exception as exc:
            log.warning("distillation stage failed: %s", exc)
        try:
            META.consider(tenant_id, run_id)
        except Exception as exc:
            log.warning("meta-agent stage failed: %s", exc)

    @staticmethod
    def _compose_final(spec: ProceduralSpec, sigma: Dict[str, Any], terminal: Dict[str, Any], report: Dict[str, Any]) -> str:
        lines: List[str] = []
        summary = str(terminal.get("summary") or terminal.get("reason") or "").strip()
        if summary:
            lines.append(summary)
        else:
            lines.append(f"Run concluded with status {terminal.get('status')}.")
        progress = [str(p) for p in (sigma.get("progress") or [])][-10:]
        if progress:
            lines.append("")
            lines.append("Verified progress:")
            for p in progress:
                lines.append(f"- {p}")
        artifacts = sigma.get("artifacts") or {}
        if isinstance(artifacts, dict) and artifacts:
            lines.append("")
            lines.append("Artifacts:")
            for name, path in list(artifacts.items())[:20]:
                lines.append(f"- {name}: {path}")
        checks = report.get("checks") or []
        if checks:
            lines.append("")
            lines.append(f"Verification score {report.get('score', 0.0):.2f} ({'passed' if report.get('passed') else 'not passed'}):")
            for c in checks[:12]:
                lines.append(f"- [{'x' if c.get('passed') else ' '}] {c.get('type')}: {str(c.get('detail'))[:200]}")
        errors = [str(e) for e in (sigma.get("errors") or [])][-6:]
        if errors:
            lines.append("")
            lines.append("Recorded error signatures:")
            for e in errors:
                lines.append(f"- {e}")
        text = "\n".join(lines)
        sanitized, _ = CLASSIFIER.sanitize(text)
        return sanitized[:20000]


ORCHESTRATOR = AgentOrchestrator()


class MetaAgent:
    SYSTEM = """You are an isolated, non-self-modifying Meta-Agent that repairs an autonomous agent's memory layers.
You never execute tasks. You only read failure diagnostics and emit minimal, scoped patch proposals.

Attribute each failure to exactly one memory component:
  "skill"      - a procedural skill is wrong, incomplete, or missing
  "wiki"       - durable environment knowledge or caveats are missing
  "state"      - the execution state schema usage is wrong
  "tool_usage" - the tool was called with incorrect arguments

Reply with exactly ONE JSON object, no prose, no markdown:
{
  "attribution": "skill" | "wiki" | "state" | "tool_usage",
  "diagnosis": "one paragraph causal analysis",
  "patches": [
    {
      "component": "skill" | "wiki",
      "operation": "create" | "revise",
      "target_name": "skill or page name",
      "category": "category",
      "summary": "what the skill does",
      "procedure": ["minimal corrected steps"],
      "preconditions": ["applicability conditions"],
      "failure_modes": ["known failure modes"],
      "tags": ["tags"],
      "body": "markdown body when component is wiki",
      "rationale": "why this patch fixes the attributed failure"
    }
  ]
}
Emit at most two patches. Keep every patch minimal and scoped to the observed failure."""

    def __init__(self, model: ModelClient, store: SQLiteStore):
        self.model = model
        self.store = store
        self._lock = threading.RLock()

    def consider(self, tenant_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            failures = TRACES.failures(tenant_id, 40)
            if run_id:
                run_failures = [f for f in failures if f["run_id"] == run_id]
                failures = run_failures or failures
            if not failures:
                return {"considered": 0, "proposed": 0, "attribution": None}
            signature_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for f in failures:
                action = f.get("action") or {}
                outcome = f.get("outcome") or {}
                key = f"{action.get('tool')}::{str(outcome.get('error') or '')[:80]}"
                signature_groups[key].append(f)
            ranked = sorted(signature_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
            top_key, group = ranked[0]
            digest_lines = [
                f"tool={ (g.get('action') or {}).get('tool') } step={g.get('step')} error={str((g.get('outcome') or {}).get('error') or '')[:220]}"
                for g in group[:12]
            ]
            skills = EM.list(tenant_id, 12)
            proposal = None
            if self.model.available:
                try:
                    out = self.model.complete(
                        [
                            {"role": "system", "content": self.SYSTEM},
                            {
                                "role": "user",
                                "content": "\n".join(
                                    [
                                        "=== RECURRING FAILURE SIGNATURE ===",
                                        top_key,
                                        f"occurrences={len(group)}",
                                        "",
                                        "=== FAILURE DIGEST ===",
                                        "\n".join(digest_lines),
                                        "",
                                        "=== CURRENT ACTIVE SKILLS ===",
                                        jdump([{"name": s.name, "category": s.category, "summary": s.summary[:200], "score": round(s.score, 3)} for s in skills])[:6000],
                                        "",
                                        "Emit the patch proposal JSON now.",
                                    ]
                                ),
                            },
                        ],
                        override={"temperature": 0.25, "max_tokens": 5000, "frequency_penalty": 0.1, "presence_penalty": 0.0},
                    )
                    proposal = DECODER.extract_json_object(out["text"])
                    TENANTS.charge_tokens(tenant_id, out["usage"]["total_tokens"])
                except Exception as exc:
                    log.warning("meta-agent proposal failed: %s", exc)
            if not isinstance(proposal, dict):
                proposal = self._heuristic(top_key, group)
            attribution = str(proposal.get("attribution") or "skill")
            if attribution not in ("skill", "wiki", "state", "tool_usage"):
                attribution = "skill"
            patches = proposal.get("patches") if isinstance(proposal.get("patches"), list) else []
            proposed = 0
            for raw in patches[:2]:
                if not isinstance(raw, dict):
                    continue
                normalized = self._normalize_patch(raw, attribution)
                if normalized is None:
                    continue
                patch_id = new_id("patch")
                self.store.execute(
                    "INSERT INTO skill_patches(patch_id, tenant_id, target_skill_id, component, diagnosis, proposal, status, gate_report, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        patch_id,
                        tenant_id,
                        normalized.get("target_skill_id"),
                        normalized["component"],
                        str(proposal.get("diagnosis") or "")[:4000],
                        jdump(normalized),
                        "proposed",
                        jdump({}),
                        iso(),
                    ),
                )
                proposed += 1
                emit_event("patch_proposed", run_id, tenant_id, {"patch_id": patch_id, "component": normalized["component"], "target": normalized.get("target_name")}, None)
                GATE.evaluate(tenant_id, patch_id)
            return {"considered": len(failures), "proposed": proposed, "attribution": attribution, "signature": top_key}

    @staticmethod
    def _heuristic(signature: str, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        tool = signature.split("::")[0]
        error = signature.split("::")[-1]
        name = f"recover_{re.sub(r'[^a-z0-9]+', '_', tool.lower())}"[:60] or "recover_generic"
        return {
            "attribution": "tool_usage",
            "diagnosis": f"The tool {tool} failed {len(group)} times with error signature '{error}'. Argument construction or precondition verification is likely incorrect.",
            "patches": [
                {
                    "component": "skill",
                    "operation": "create",
                    "target_name": name,
                    "category": "recovery",
                    "summary": f"Deterministic recovery procedure for repeated {tool} failures with signature '{error}'.",
                    "procedure": [
                        f"Before calling {tool}, verify every precondition with a read-only tool (workspace.list_dir or workspace.read_file).",
                        f"Record the exact argument set in state.scratch prior to invoking {tool}.",
                        f"If {tool} fails, append the error signature to state.errors and do not retry with identical arguments.",
                        "Select an alternative tool path or reduce the scope of the operation, then re-verify.",
                    ],
                    "preconditions": [f"A prior invocation of {tool} failed with '{error}'."],
                    "failure_modes": [f"{tool} repeatedly failing with '{error}'"],
                    "tags": ["recovery", tool.split(".")[0] if "." in tool else tool],
                    "rationale": "Prevents identical retry loops and forces precondition verification.",
                }
            ],
        }

    @staticmethod
    def _normalize_patch(raw: Dict[str, Any], attribution: str) -> Optional[Dict[str, Any]]:
        component = str(raw.get("component") or ("wiki" if attribution == "wiki" else "skill")).lower()
        if component not in ("skill", "wiki"):
            component = "skill"
        target_name = str(raw.get("target_name") or "").strip()[:120]
        if not target_name:
            return None
        if component == "skill":
            procedure = [str(p)[:400] for p in (raw.get("procedure") or []) if p][:24]
            summary = str(raw.get("summary") or "").strip()[:800]
            if not procedure or not summary:
                return None
            return {
                "component": "skill",
                "operation": str(raw.get("operation") or "create").lower(),
                "target_name": target_name,
                "category": str(raw.get("category") or "general")[:60],
                "summary": summary,
                "procedure": procedure,
                "preconditions": [str(p)[:300] for p in (raw.get("preconditions") or [])][:12],
                "failure_modes": [str(p)[:300] for p in (raw.get("failure_modes") or [])][:12],
                "tags": [str(t)[:40] for t in (raw.get("tags") or [])][:12],
                "rationale": str(raw.get("rationale") or "")[:800],
            }
        body = str(raw.get("body") or "").strip()
        if not body:
            return None
        return {
            "component": "wiki",
            "operation": str(raw.get("operation") or "revise").lower(),
            "target_name": target_name,
            "category": str(raw.get("category") or "general")[:60],
            "body": body[:24000],
            "rationale": str(raw.get("rationale") or "")[:800],
        }


class ValidationGate:
    def __init__(self, store: SQLiteStore):
        self.store = store
        self._lock = threading.RLock()

    def _diagnostics(self, tenant_id: str) -> List[Dict[str, Any]]:
        rows = self.store.query("SELECT * FROM diagnostic_tasks WHERE tenant_id = ? ORDER BY created_at ASC", (tenant_id,))
        if rows:
            return [
                {
                    "task_id": r["task_id"],
                    "name": r["name"],
                    "spec": jload(r["spec"], {}) or {},
                    "verifier": jload(r["verifier"], {}) or {},
                    "baseline_score": float(r["baseline_score"]),
                }
                for r in rows
            ]
        return self._seed(tenant_id)

    def _seed(self, tenant_id: str) -> List[Dict[str, Any]]:
        seeds = [
            {
                "name": "schema_conformance",
                "spec": {"objective": "Emit a state patch and single tool action conforming to the runtime contract."},
                "verifier": {"type": "structure", "required_keys": ["procedure", "summary"], "min_procedure_steps": 2},
            },
            {
                "name": "determinism_and_scope",
                "spec": {"objective": "Skill procedures must be deterministic, scoped, and free of destructive operations."},
                "verifier": {"type": "safety", "forbidden": ["rm -rf", "mkfs", "sudo", "curl ", "wget ", "chmod 777"]},
            },
            {
                "name": "actionability",
                "spec": {"objective": "Each procedure step must reference a registered tool or a concrete verifiable check."},
                "verifier": {"type": "actionability", "min_tool_references": 1},
            },
            {
                "name": "no_regression_in_library",
                "spec": {"objective": "A patch must not duplicate or contradict an existing high-scoring skill."},
                "verifier": {"type": "no_duplicate", "similarity_threshold": 0.94},
            },
        ]
        out: List[Dict[str, Any]] = []
        for s in seeds:
            task_id = new_id("diag")
            self.store.execute(
                "INSERT INTO diagnostic_tasks(task_id, tenant_id, name, spec, verifier, baseline_score, created_at) VALUES(?,?,?,?,?,?,?)",
                (task_id, tenant_id, s["name"], jdump(s["spec"]), jdump(s["verifier"]), 1.0, iso()),
            )
            out.append({"task_id": task_id, "name": s["name"], "spec": s["spec"], "verifier": s["verifier"], "baseline_score": 1.0})
        return out

    def _run_check(self, tenant_id: str, task: Dict[str, Any], proposal: Dict[str, Any]) -> Dict[str, Any]:
        verifier = task["verifier"]
        vtype = str(verifier.get("type") or "")
        component = proposal.get("component")
        if vtype == "structure":
            if component == "skill":
                ok = bool(proposal.get("summary")) and len(proposal.get("procedure") or []) >= int(verifier.get("min_procedure_steps") or 2)
                detail = f"summary={bool(proposal.get('summary'))} steps={len(proposal.get('procedure') or [])}"
            else:
                ok = len(str(proposal.get("body") or "")) >= 40
                detail = f"body_len={len(str(proposal.get('body') or ''))}"
            return {"task": task["name"], "passed": ok, "detail": detail}
        if vtype == "safety":
            blob = jdump(proposal).lower()
            forbidden = [f for f in (verifier.get("forbidden") or []) if str(f).lower() in blob]
            verdict = CLASSIFIER.classify(jdump(proposal))
            ok = not forbidden and verdict["severity"] != "block"
            return {"task": task["name"], "passed": ok, "detail": f"forbidden={forbidden} classifier={verdict['severity']}"}
        if vtype == "actionability":
            if component != "skill":
                return {"task": task["name"], "passed": True, "detail": "not applicable to wiki patches"}
            blob = " ".join(str(s) for s in (proposal.get("procedure") or []))
            refs = sum(1 for name in TOOLS.names() if name in blob or name.split(".")[-1] in blob)
            concrete = sum(1 for s in (proposal.get("procedure") or []) if re.search(r"(verify|check|read|write|append|replace|record|compare|assert|list)", str(s), re.IGNORECASE))
            ok = (refs >= int(verifier.get("min_tool_references") or 1)) or concrete >= 2
            return {"task": task["name"], "passed": ok, "detail": f"tool_refs={refs} concrete_steps={concrete}"}
        if vtype == "no_duplicate":
            if component != "skill":
                return {"task": task["name"], "passed": True, "detail": "not applicable to wiki patches"}
            threshold = float(verifier.get("similarity_threshold") or 0.94)
            text = " ".join([str(proposal.get("summary") or ""), " ".join(str(x) for x in (proposal.get("procedure") or []))])
            qvec = EMBEDDER.embed(text)
            worst = 0.0
            worst_name = ""
            for skill in EM.list(tenant_id, 120):
                if skill.name == proposal.get("target_name"):
                    continue
                sim = EMBEDDER.cosine(qvec, EMBEDDER.embed(" ".join([skill.summary, " ".join(str(x) for x in skill.procedure)])))
                if sim > worst:
                    worst = sim
                    worst_name = skill.name
            ok = worst < threshold
            return {"task": task["name"], "passed": ok, "detail": f"max_similarity={worst:.4f} vs '{worst_name}' threshold={threshold}"}
        return {"task": task["name"], "passed": True, "detail": "unknown verifier treated as neutral"}

    def evaluate(self, tenant_id: str, patch_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self.store.one("SELECT * FROM skill_patches WHERE patch_id = ? AND tenant_id = ?", (patch_id, tenant_id))
            if row is None:
                raise ValidationError(f"patch {patch_id} not found")
            if row["status"] != "proposed":
                return {"patch_id": patch_id, "status": row["status"], "gate_report": jload(row["gate_report"], {})}
            proposal = jload(row["proposal"], {}) or {}
            tasks = self._diagnostics(tenant_id)
            checks = [self._run_check(tenant_id, t, proposal) for t in tasks]
            passed_count = sum(1 for c in checks if c["passed"])
            score = passed_count / max(1, len(checks))
            accepted = all(c["passed"] for c in checks)
            report = {
                "checks": checks,
                "score": score,
                "accepted": accepted,
                "evaluated_at": iso(),
                "baseline": 1.0,
                "regression": (not accepted),
            }
            applied: Dict[str, Any] = {}
            status = "rejected"
            if accepted:
                try:
                    applied = self._apply(tenant_id, proposal)
                    status = "applied"
                except Exception as exc:
                    status = "rollback"
                    report["apply_error"] = str(exc)[:800]
                    log.warning("patch apply failed, rolled back: %s", exc)
            report["applied"] = applied
            self.store.execute(
                "UPDATE skill_patches SET status = ?, gate_report = ?, decided_at = ? WHERE patch_id = ?",
                (status, jdump(report), iso(), patch_id),
            )
            audit(tenant_id, None, "validation_gate", f"patch_{status}", {"patch_id": patch_id, "score": score}, accepted)
            emit_event("patch_decision", None, tenant_id, {"patch_id": patch_id, "status": status, "score": score, "checks": checks}, None)
            return {"patch_id": patch_id, "status": status, "gate_report": report}

    def _apply(self, tenant_id: str, proposal: Dict[str, Any]) -> Dict[str, Any]:
        component = proposal.get("component")
        if component == "skill":
            skill = EM.upsert(
                tenant_id,
                proposal["target_name"],
                proposal["summary"],
                proposal["procedure"],
                proposal.get("preconditions"),
                proposal.get("failure_modes"),
                proposal.get("tags"),
                proposal.get("category", "general"),
            )
            return {"component": "skill", "skill_id": skill.skill_id, "name": skill.name, "version": skill.version}
        page = WIKI.upsert(tenant_id, proposal["target_name"], proposal["body"], proposal.get("category", "general"))
        return {"component": "wiki", **page}

    def rollback(self, tenant_id: str, patch_id: str) -> Dict[str, Any]:
        row = self.store.one("SELECT * FROM skill_patches WHERE patch_id = ? AND tenant_id = ?", (patch_id, tenant_id))
        if row is None:
            raise ValidationError("patch not found")
        report = jload(row["gate_report"], {}) or {}
        applied = report.get("applied") or {}
        if applied.get("component") == "skill" and applied.get("skill_id"):
            EM.retire(tenant_id, applied["skill_id"])
        self.store.execute("UPDATE skill_patches SET status = 'rollback', decided_at = ? WHERE patch_id = ?", (iso(), patch_id))
        audit(tenant_id, None, "validation_gate", "patch_rollback", {"patch_id": patch_id}, True)
        return {"patch_id": patch_id, "status": "rollback"}


GATE = ValidationGate(STORE)
META = MetaAgent(MODEL, STORE)


CHAT_SYSTEM_PROMPT = "You are a helpful assistant. Be concise and accurate. Answer in plain text without markdown unless asked."

PLANNER_SYSTEM = """You are the intake planner of an autonomous agent runtime. You convert a user request into an executable procedural specification.

Reply with exactly ONE JSON object, no prose, no markdown:
{
  "mode": "chat" | "agent",
  "objective": "single imperative sentence describing the complete goal",
  "success_criteria": ["objectively verifiable completion conditions"],
  "constraints": ["hard constraints the agent must never violate"],
  "max_steps": 60,
  "verifiers": [{"type": "all_criteria_verified"}],
  "reply": "direct plain-text answer when mode is chat, otherwise empty string"
}

Choose "chat" only for pure conversation, greetings, or a question you can fully answer in one reply with no tools.
Choose "agent" whenever the request requires multi-step work, file creation, computation, research, iteration, or verification.
Supported verifier types: all_criteria_verified, file_exists, file_contains, file_min_lines, state_key_truthy, state_key_equals, no_errors, python_assert."""


class ChatService:
    def __init__(self, model: ModelClient):
        self.model = model

    def plan(self, tenant_id: str, message: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        recent = history[-6:]
        context_lines = [f"{m['role']}: {str(m['content'])[:600]}" for m in recent]
        if not self.model.available:
            return {
                "mode": "agent",
                "objective": message.strip()[:2000] or "Assist the user.",
                "success_criteria": ["The user request is fully satisfied and every claim is verified."],
                "constraints": ["Do not fabricate results.", "Verify each success criterion before finishing."],
                "max_steps": 60,
                "verifiers": [{"type": "all_criteria_verified"}],
                "reply": "",
            }
        try:
            out = self.model.complete(
                [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                "=== RECENT CONVERSATION ===",
                                "\n".join(context_lines) if context_lines else "(none)",
                                "",
                                "=== CURRENT USER REQUEST ===",
                                message[:12000],
                                "",
                                "=== AVAILABLE TOOLS ===",
                                TOOLS.describe(),
                                "",
                                "Emit the specification JSON now.",
                            ]
                        ),
                    },
                ],
                override={"temperature": 0.3, "max_tokens": 4000, "frequency_penalty": 0.1, "presence_penalty": 0.0},
            )
            TENANTS.charge_tokens(tenant_id, out["usage"]["total_tokens"])
            parsed = DECODER.extract_json_object(out["text"])
        except Exception as exc:
            log.warning("planner failed: %s", exc)
            parsed = None
        if not isinstance(parsed, dict):
            return {
                "mode": "agent",
                "objective": message.strip()[:2000] or "Assist the user.",
                "success_criteria": ["The user request is fully satisfied and every claim is verified."],
                "constraints": ["Do not fabricate results."],
                "max_steps": 60,
                "verifiers": [{"type": "all_criteria_verified"}],
                "reply": "",
            }
        mode = str(parsed.get("mode") or "agent").lower()
        if mode not in ("chat", "agent"):
            mode = "agent"
        criteria = [str(c)[:400] for c in (parsed.get("success_criteria") or []) if c][:12]
        constraints = [str(c)[:400] for c in (parsed.get("constraints") or []) if c][:12]
        verifiers: List[Dict[str, Any]] = []
        for v in (parsed.get("verifiers") or [])[:12]:
            if isinstance(v, dict) and v.get("type"):
                verifiers.append({str(k): v[k] for k in v})
        if not verifiers:
            verifiers = [{"type": "all_criteria_verified"}]
        return {
            "mode": mode,
            "objective": str(parsed.get("objective") or message)[:4000],
            "success_criteria": criteria or ["The user request is fully satisfied and verified."],
            "constraints": constraints,
            "max_steps": int(clamp(float(parsed.get("max_steps") or 60), 1, 2000)),
            "verifiers": verifiers,
            "reply": str(parsed.get("reply") or ""),
        }

    def chat_stream(self, tenant_id: str, conversation_id: str, message: str, history: List[Dict[str, Any]]) -> Iterable[str]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        for m in history[-16:]:
            role = m.get("role")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": str(m.get("content") or "")[:8000]})
        messages.append({"role": "user", "content": message[:20000]})
        chunks: List[str] = []
        queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

        def collector(delta: str) -> None:
            chunks.append(delta)

        result = self.model.stream(messages, on_delta=collector)
        TENANTS.charge_tokens(tenant_id, result["usage"]["total_tokens"])
        text = result["text"]
        sanitized, verdict = CLASSIFIER.sanitize(text)
        MESSAGES.add(tenant_id, conversation_id, "assistant", sanitized, {"mode": "chat", "classifier": verdict, "usage": result["usage"]})
        yield sanitized


CHAT = ChatService(MODEL)


class StartRunRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=20000)
    success_criteria: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    allowed_tools: Optional[List[str]] = None
    max_steps: int = Field(default=DEFAULT_STEP_BUDGET, ge=1, le=5000)
    verifiers: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=100000)
    conversation_id: Optional[str] = None
    force_agent: bool = False
    max_steps: Optional[int] = Field(default=None, ge=1, le=5000)


class SkillRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    summary: str = Field(min_length=1, max_length=4000)
    procedure: List[str] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    category: str = "general"


class WikiRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=200000)
    category: str = "general"


class OptimizeRequest(BaseModel):
    run_id: Optional[str] = None
    learning_rate: float = Field(default=0.04, gt=0.0, le=1.0)
    epochs: int = Field(default=2, ge=1, le=20)
    limit: int = Field(default=400, ge=1, le=5000)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class TenantRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=160)
    api_key: str = Field(min_length=8, max_length=256)
    token_budget: int = Field(default=DEFAULT_TOKEN_BUDGET, ge=1000)


app = FastAPI(title="Autonomous Agent Runtime", version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("AGENT_CORS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def resolve_tenant(
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-Id"),
    x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key"),
) -> Tenant:
    try:
        return TENANTS.authenticate(x_tenant_id, x_api_key)
    except SecurityError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


async def require_admin(x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")) -> bool:
    if not x_admin_token or not hmac.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="admin token required")
    return True


@app.exception_handler(SecurityError)
async def security_handler(request: Request, exc: SecurityError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"error": "security_error", "detail": str(exc)})


@app.exception_handler(ValidationError)
async def validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "validation_error", "detail": str(exc)})


@app.exception_handler(BudgetExceeded)
async def budget_handler(request: Request, exc: BudgetExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"error": "budget_exceeded", "detail": str(exc)})


@app.exception_handler(ToolDenied)
async def tool_handler(request: Request, exc: ToolDenied) -> JSONResponse:
    return JSONResponse(status_code=403, content={"error": "tool_denied", "detail": str(exc)})


@app.on_event("startup")
async def on_startup() -> None:
    BUS.bind_loop(asyncio.get_running_loop())
    log.info("agent runtime starting; home=%s model=%s api_key_present=%s", BASE_DIR, MODEL_NAME, bool(MODEL_API_KEY))
    recovered = await asyncio.get_running_loop().run_in_executor(None, ORCHESTRATOR.recover)
    log.info("recovered %d interrupted runs from durable checkpoints", recovered)
    asyncio.create_task(_janitor())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    log.info("agent runtime shutting down; checkpointing active runs")
    await asyncio.get_running_loop().run_in_executor(None, ORCHESTRATOR.shutdown)


async def _janitor() -> None:
    while True:
        try:
            await asyncio.sleep(45.0)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _janitor_pass)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log.warning("janitor pass failed: %s", exc)


def _janitor_pass() -> None:
    now = utcnow()
    rows = STORE.query("SELECT run_id, tenant_id, spec, conversation_id, lease_owner, lease_expires_at, status FROM runs WHERE status = ?", (RunStatus.RUNNING.value,))
    for r in rows:
        if ORCHESTRATOR._handle(r["run_id"]) is not None:
            continue
        exp = r["lease_expires_at"]
        expired = True
        if exp:
            try:
                expired = datetime.fromisoformat(exp) <= now
            except Exception:
                expired = True
        if not expired:
            continue
        try:
            spec = ProceduralSpec.from_dict(jload(r["spec"], {}) or {})
            ORCHESTRATOR.start(r["tenant_id"], r["conversation_id"], spec, resume_run_id=r["run_id"])
            log.info("janitor resumed orphaned run %s", r["run_id"])
        except Exception as exc:
            log.warning("janitor resume failed for %s: %s", r["run_id"], exc)
    STORE.execute("DELETE FROM events WHERE created_at < ?", ((now - timedelta(days=7)).isoformat(),))


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    row = STORE.one("SELECT COUNT(*) AS c FROM runs")
    return {
        "status": "ok",
        "time": iso(),
        "model": MODEL_NAME,
        "model_configured": MODEL.available,
        "base_url": MODEL_BASE_URL,
        "home": str(BASE_DIR),
        "runs_total": int(row["c"]) if row else 0,
        "active_runs": ORCHESTRATOR.active_runs(),
        "system1_hz": SYSTEM1_HZ,
        "system2_hz": SYSTEM2_HZ,
        "tools": TOOLS.names(),
        "tokens": {"prompt": MODEL.total_prompt_tokens, "completion": MODEL.total_completion_tokens},
    }


@app.get("/api/config")
async def config(tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    return {
        "tenant_id": tenant.tenant_id,
        "name": tenant.name,
        "token_budget": tenant.token_budget,
        "tokens_used": tenant.tokens_used,
        "allowed_tools": tenant.allowed_tools,
        "model": MODEL_NAME,
        "model_configured": MODEL.available,
        "max_steps_default": DEFAULT_STEP_BUDGET,
        "verifier_types": [
            "all_criteria_verified",
            "file_exists",
            "file_contains",
            "file_min_lines",
            "state_key_truthy",
            "state_key_equals",
            "no_errors",
            "python_assert",
        ],
    }


@app.get("/api/conversations")
async def list_conversations(tenant: Tenant = Depends(resolve_tenant), limit: int = 100) -> Dict[str, Any]:
    return {"conversations": MESSAGES.conversations(tenant.tenant_id, min(500, max(1, limit)))}


@app.post("/api/conversations")
async def create_conversation(tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    cid = MESSAGES.ensure_conversation(tenant.tenant_id, None, "New conversation")
    return {"conversation_id": cid}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    row = STORE.one("SELECT * FROM conversations WHERE conversation_id = ? AND tenant_id = ?", (conversation_id, tenant.tenant_id))
    if row is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {
        "conversation_id": conversation_id,
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": MESSAGES.history(tenant.tenant_id, conversation_id),
    }


@app.patch("/api/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, payload: RenameRequest, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    MESSAGES.rename(tenant.tenant_id, conversation_id, payload.title)
    return {"conversation_id": conversation_id, "title": payload.title}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    MESSAGES.delete(tenant.tenant_id, conversation_id)
    return {"deleted": True, "conversation_id": conversation_id}


@app.post("/api/chat")
async def chat(payload: ChatRequest, tenant: Tenant = Depends(resolve_tenant)) -> StreamingResponse:
    tenant_id = tenant.tenant_id
    conversation_id = MESSAGES.ensure_conversation(tenant_id, payload.conversation_id, payload.message[:80])
    history = MESSAGES.history(tenant_id, conversation_id)
    user_msg = MESSAGES.add(tenant_id, conversation_id, "user", payload.message, {})
    if len(history) == 0:
        MESSAGES.rename(tenant_id, conversation_id, payload.message[:80])
    loop = asyncio.get_running_loop()

    async def event_stream() -> Iterable[bytes]:
        def sse(kind: str, data: Dict[str, Any]) -> bytes:
            return f"event: {kind}\ndata: {jdump(data)}\n\n".encode("utf-8")

        yield sse("conversation", {"conversation_id": conversation_id, "message": user_msg})
        if not MODEL.available:
            detail = "MODULAR_API_KEY is not configured on the server; the model backend is unavailable."
            MESSAGES.add(tenant_id, conversation_id, "assistant", detail, {"error": "model_unconfigured"})
            yield sse("error", {"detail": detail})
            yield sse("done", {"conversation_id": conversation_id})
            return
        try:
            plan = await loop.run_in_executor(None, CHAT.plan, tenant_id, payload.message, history)
        except Exception as exc:
            log.warning("planning failed: %s", exc)
            plan = {
                "mode": "agent",
                "objective": payload.message[:4000],
                "success_criteria": ["The user request is fully satisfied and verified."],
                "constraints": [],
                "max_steps": 60,
                "verifiers": [{"type": "all_criteria_verified"}],
                "reply": "",
            }
        mode = "agent" if payload.force_agent else plan["mode"]
        yield sse("plan", {"mode": mode, "objective": plan["objective"], "success_criteria": plan["success_criteria"], "constraints": plan["constraints"], "max_steps": plan["max_steps"]})

        if mode == "chat":
            reply = str(plan.get("reply") or "").strip()
            if reply:
                sanitized, verdict = CLASSIFIER.sanitize(reply)
                MESSAGES.add(tenant_id, conversation_id, "assistant", sanitized, {"mode": "chat", "classifier": verdict})
                for i in range(0, len(sanitized), 320):
                    yield sse("delta", {"delta": sanitized[i : i + 320]})
                    await asyncio.sleep(0)
                yield sse("message", {"role": "assistant", "content": sanitized})
                yield sse("done", {"conversation_id": conversation_id})
                return
            queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

            def push(delta: str) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, delta)

            def worker() -> Dict[str, Any]:
                msgs: List[Dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
                for m in history[-16:]:
                    if m.get("role") in ("user", "assistant"):
                        msgs.append({"role": str(m["role"]), "content": str(m.get("content") or "")[:8000]})
                msgs.append({"role": "user", "content": payload.message[:20000]})
                try:
                    return MODEL.stream(msgs, on_delta=push)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            task = loop.run_in_executor(None, worker)
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield sse("delta", {"delta": item})
            try:
                result = await task
            except Exception as exc:
                detail = f"model error: {exc}"[:800]
                MESSAGES.add(tenant_id, conversation_id, "assistant", detail, {"error": "model_error"})
                yield sse("error", {"detail": detail})
                yield sse("done", {"conversation_id": conversation_id})
                return
            text = result["text"]
            sanitized, verdict = CLASSIFIER.sanitize(text)
            try:
                TENANTS.charge_tokens(tenant_id, int(result["usage"].get("total_tokens") or 0))
            except BudgetExceeded as exc:
                yield sse("error", {"detail": str(exc)})
            MESSAGES.add(tenant_id, conversation_id, "assistant", sanitized, {"mode": "chat", "classifier": verdict, "usage": result["usage"]})
            yield sse("message", {"role": "assistant", "content": sanitized})
            yield sse("done", {"conversation_id": conversation_id})
            return

        spec = ProceduralSpec.build(
            tenant_id=tenant_id,
            objective=plan["objective"],
            success_criteria=plan["success_criteria"],
            constraints=plan["constraints"],
            allowed_tools=tenant.allowed_tools,
            max_steps=int(payload.max_steps or plan["max_steps"]),
            verifiers=plan["verifiers"],
            metadata={"conversation_id": conversation_id, "origin": "chat"},
        )
        queue = BUS.subscribe(f"conv:{conversation_id}")
        try:
            run_id = await loop.run_in_executor(None, ORCHESTRATOR.start, tenant_id, conversation_id, spec, None)
        except Exception as exc:
            BUS.unsubscribe(f"conv:{conversation_id}", queue)
            detail = f"failed to start run: {exc}"[:600]
            MESSAGES.add(tenant_id, conversation_id, "assistant", detail, {"error": "run_start_failed"})
            yield sse("error", {"detail": detail})
            yield sse("done", {"conversation_id": conversation_id})
            return
        yield sse("run_started", {"run_id": run_id, "objective": spec.objective, "max_steps": spec.max_steps})
        try:
            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=25.0)
                except asyncio.TimeoutError:
                    row = RUNS.get(run_id, tenant_id)
                    status = row["status"] if row else "unknown"
                    yield sse("heartbeat", {"run_id": run_id, "status": status, "step": int(row["step"]) if row else 0})
                    if status in (RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value, RunStatus.ERROR.value):
                        break
                    continue
                if evt.get("run_id") != run_id:
                    continue
                yield sse(evt["kind"], {"run_id": run_id, "seq": evt["seq"], **(evt.get("payload") or {})})
                if evt["kind"] in ("run_finished", "run_error", "run_cancelled"):
                    break
        finally:
            BUS.unsubscribe(f"conv:{conversation_id}", queue)
        yield sse("done", {"conversation_id": conversation_id, "run_id": run_id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/api/runs")
async def create_run(payload: StartRunRequest, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    allowed = payload.allowed_tools or tenant.allowed_tools
    invalid = [t for t in allowed if t not in TOOLS.names()]
    if invalid:
        raise HTTPException(status_code=400, detail=f"unknown tools requested: {invalid}")
    denied = [t for t in allowed if t not in tenant.allowed_tools]
    if denied:
        raise HTTPException(status_code=403, detail=f"tools not authorized for tenant: {denied}")
    conversation_id = MESSAGES.ensure_conversation(tenant.tenant_id, payload.conversation_id, payload.objective[:80]) if payload.conversation_id else None
    spec = ProceduralSpec.build(
        tenant_id=tenant.tenant_id,
        objective=payload.objective,
        success_criteria=payload.success_criteria,
        constraints=payload.constraints,
        allowed_tools=allowed,
        max_steps=payload.max_steps,
        verifiers=payload.verifiers,
        metadata=payload.metadata,
    )
    loop = asyncio.get_running_loop()
    run_id = await loop.run_in_executor(None, ORCHESTRATOR.start, tenant.tenant_id, conversation_id, spec, None)
    return {"run_id": run_id, "status": RunStatus.RUNNING.value, "spec": spec.to_dict()}


@app.get("/api/runs")
async def list_runs(tenant: Tenant = Depends(resolve_tenant), limit: int = 100, status: Optional[str] = None) -> Dict[str, Any]:
    return {"runs": RUNS.list(tenant.tenant_id, min(500, max(1, limit)), status), "active": ORCHESTRATOR.active_runs()}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    row = RUNS.get(run_id, tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    ckpt = CHECKPOINTS.latest(run_id)
    return {
        "run_id": run_id,
        "status": row["status"],
        "step": int(row["step"]),
        "spec": jload(row["spec"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "finished_at": row["finished_at"],
        "tokens_used": int(row["tokens_used"]),
        "wall_ms": int(row["wall_ms"]),
        "verdict": jload(row["verdict"], None),
        "terminal_state": jload(row["terminal_state"], None),
        "error": row["error"],
        "resume_count": int(row["resume_count"]),
        "state": ckpt["sigma"] if ckpt else None,
        "observation": ckpt["observation"] if ckpt else None,
        "checkpoint_integrity": ckpt["integrity_ok"] if ckpt else None,
        "working_memory": WM.load(tenant.tenant_id, run_id),
        "cognition": COGNITION.latest(run_id),
    }


@app.get("/api/runs/{run_id}/traces")
async def run_traces(run_id: str, tenant: Tenant = Depends(resolve_tenant), limit: int = 200) -> Dict[str, Any]:
    row = RUNS.get(run_id, tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "traces": TRACES.for_run(run_id, min(2000, max(1, limit)))}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, tenant: Tenant = Depends(resolve_tenant), after: int = 0, limit: int = 500) -> Dict[str, Any]:
    row = RUNS.get(run_id, tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    rows = STORE.query(
        "SELECT event_id, kind, payload, created_at, seq FROM events WHERE run_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
        (run_id, int(after), min(2000, max(1, limit))),
    )
    return {
        "run_id": run_id,
        "events": [{"event_id": r["event_id"], "kind": r["kind"], "payload": jload(r["payload"], {}), "created_at": r["created_at"], "seq": int(r["seq"])} for r in rows],
    }


@app.get("/api/runs/{run_id}/checkpoints")
async def run_checkpoints(run_id: str, tenant: Tenant = Depends(resolve_tenant), limit: int = 50) -> Dict[str, Any]:
    row = RUNS.get(run_id, tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    rows = STORE.query(
        "SELECT checkpoint_id, step, node, digest, created_at FROM checkpoints WHERE run_id = ? ORDER BY step DESC, created_at DESC LIMIT ?",
        (run_id, min(500, max(1, limit))),
    )
    return {"run_id": run_id, "checkpoints": [dict(r) for r in rows], "latest": CHECKPOINTS.latest(run_id)}


@app.post("/api/runs/{run_id}/pause")
async def pause_run(run_id: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    if RUNS.get(run_id, tenant.tenant_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "paused": ORCHESTRATOR.pause(run_id)}


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    row = RUNS.get(run_id, tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    if ORCHESTRATOR.resume_paused(run_id):
        return {"run_id": run_id, "resumed": True, "mode": "unpaused"}
    spec = ProceduralSpec.from_dict(jload(row["spec"], {}) or {})
    loop = asyncio.get_running_loop()
    new_run = await loop.run_in_executor(None, ORCHESTRATOR.start, tenant.tenant_id, row["conversation_id"], spec, run_id)
    return {"run_id": new_run, "resumed": True, "mode": "restarted_from_checkpoint"}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    if RUNS.get(run_id, tenant.tenant_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "cancelled": ORCHESTRATOR.cancel(run_id)}


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str, tenant: Tenant = Depends(resolve_tenant)) -> StreamingResponse:
    row = RUNS.get(run_id, tenant.tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")

    async def gen() -> Iterable[bytes]:
        queue = BUS.subscribe(f"run:{run_id}")
        try:
            snapshot = CHECKPOINTS.latest(run_id)
            yield f"event: snapshot\ndata: {jdump({'run_id': run_id, 'status': row['status'], 'step': int(row['step']), 'state': snapshot['sigma'] if snapshot else None})}\n\n".encode("utf-8")
            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=25.0)
                except asyncio.TimeoutError:
                    current = RUNS.get(run_id, tenant.tenant_id)
                    status = current["status"] if current else "unknown"
                    yield f"event: heartbeat\ndata: {jdump({'run_id': run_id, 'status': status})}\n\n".encode("utf-8")
                    if status in (RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value, RunStatus.ERROR.value):
                        break
                    continue
                yield f"event: {evt['kind']}\ndata: {jdump({'run_id': run_id, 'seq': evt['seq'], **(evt.get('payload') or {})})}\n\n".encode("utf-8")
                if evt["kind"] in ("run_finished", "run_error", "run_cancelled"):
                    break
        finally:
            BUS.unsubscribe(f"run:{run_id}", queue)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    tenant_id = websocket.query_params.get("tenant_id") or "public"
    api_key = websocket.query_params.get("api_key")
    try:
        tenant = TENANTS.authenticate(tenant_id, api_key)
    except SecurityError as exc:
        await websocket.send_text(jdump({"kind": "error", "detail": str(exc)}))
        await websocket.close(code=4401)
        return
    topic = "global"
    run_id = websocket.query_params.get("run_id")
    conversation_id = websocket.query_params.get("conversation_id")
    if run_id:
        topic = f"run:{run_id}"
    elif conversation_id:
        topic = f"conv:{conversation_id}"
    queue = BUS.subscribe(topic)
    await websocket.send_text(jdump({"kind": "connected", "topic": topic, "tenant_id": tenant.tenant_id, "time": iso()}))

    async def pump() -> None:
        while True:
            evt = await queue.get()
            if evt.get("tenant_id") and evt.get("tenant_id") != tenant.tenant_id:
                continue
            await websocket.send_text(jdump(evt))

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                data = {"kind": "ping"}
            kind = str(data.get("kind") or "ping")
            if kind == "ping":
                await websocket.send_text(jdump({"kind": "pong", "time": iso()}))
            elif kind == "cancel" and data.get("run_id"):
                ORCHESTRATOR.cancel(str(data["run_id"]))
                await websocket.send_text(jdump({"kind": "ack", "action": "cancel", "run_id": data["run_id"]}))
            elif kind == "pause" and data.get("run_id"):
                ORCHESTRATOR.pause(str(data["run_id"]))
                await websocket.send_text(jdump({"kind": "ack", "action": "pause", "run_id": data["run_id"]}))
            elif kind == "resume" and data.get("run_id"):
                ORCHESTRATOR.resume_paused(str(data["run_id"]))
                await websocket.send_text(jdump({"kind": "ack", "action": "resume", "run_id": data["run_id"]}))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("websocket closed: %s", exc)
    finally:
        pump_task.cancel()
        BUS.unsubscribe(topic, queue)


@app.get("/api/skills")
async def list_skills(tenant: Tenant = Depends(resolve_tenant), limit: int = 200, query: Optional[str] = None) -> Dict[str, Any]:
    if query:
        found = EM.search(tenant.tenant_id, query, min(50, max(1, limit)))
        return {"skills": [{**s.to_dict(), "relevance": sc} for s, sc in found]}
    return {"skills": [s.to_dict() for s in EM.list(tenant.tenant_id, min(1000, max(1, limit)))]}


@app.post("/api/skills")
async def upsert_skill(payload: SkillRequest, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    skill = EM.upsert(
        tenant.tenant_id,
        payload.name,
        payload.summary,
        payload.procedure,
        payload.preconditions,
        payload.failure_modes,
        payload.tags,
        payload.category,
    )
    return {"skill": skill.to_dict()}


@app.delete("/api/skills/{skill_id}")
async def retire_skill(skill_id: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    if EM.get(tenant.tenant_id, skill_id) is None:
        raise HTTPException(status_code=404, detail="skill not found")
    EM.retire(tenant.tenant_id, skill_id)
    return {"skill_id": skill_id, "retired": True}


@app.get("/api/wiki")
async def list_wiki(tenant: Tenant = Depends(resolve_tenant), limit: int = 100, query: Optional[str] = None) -> Dict[str, Any]:
    if query:
        return {"pages": WIKI.search(tenant.tenant_id, query, min(50, max(1, limit)))}
    return {"pages": WIKI.list_pages(tenant.tenant_id, min(500, max(1, limit)))}


@app.post("/api/wiki")
async def upsert_wiki(payload: WikiRequest, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    return {"page": WIKI.upsert(tenant.tenant_id, payload.title, payload.body, payload.category)}


@app.get("/api/wiki/{slug}")
async def get_wiki(slug: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    page = WIKI.get_page(tenant.tenant_id, slug)
    if page is None:
        raise HTTPException(status_code=404, detail="page not found")
    revisions = STORE.query(
        "SELECT revision_id, version, diff, created_at FROM wiki_revisions WHERE page_id = ? ORDER BY version DESC LIMIT 30",
        (page["page_id"],),
    )
    return {"page": page, "revisions": [dict(r) for r in revisions]}


@app.get("/api/patches")
async def list_patches(tenant: Tenant = Depends(resolve_tenant), limit: int = 100, status: Optional[str] = None) -> Dict[str, Any]:
    if status:
        rows = STORE.query(
            "SELECT * FROM skill_patches WHERE tenant_id = ? AND status = ? ORDER BY created_at DESC LIMIT ?",
            (tenant.tenant_id, status, min(500, max(1, limit))),
        )
    else:
        rows = STORE.query(
            "SELECT * FROM skill_patches WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
            (tenant.tenant_id, min(500, max(1, limit))),
        )
    return {
        "patches": [
            {
                "patch_id": r["patch_id"],
                "component": r["component"],
                "status": r["status"],
                "diagnosis": r["diagnosis"],
                "proposal": jload(r["proposal"], {}),
                "gate_report": jload(r["gate_report"], {}),
                "created_at": r["created_at"],
                "decided_at": r["decided_at"],
            }
            for r in rows
        ]
    }


@app.post("/api/patches/evaluate")
async def meta_evaluate(tenant: Tenant = Depends(resolve_tenant), run_id: Optional[str] = None) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, META.consider, tenant.tenant_id, run_id)
    return result


@app.post("/api/patches/{patch_id}/gate")
async def gate_patch(patch_id: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, GATE.evaluate, tenant.tenant_id, patch_id)


@app.post("/api/patches/{patch_id}/rollback")
async def rollback_patch(patch_id: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, GATE.rollback, tenant.tenant_id, patch_id)


@app.get("/api/reflections")
async def list_reflections(tenant: Tenant = Depends(resolve_tenant), limit: int = 50, run_id: Optional[str] = None) -> Dict[str, Any]:
    if run_id:
        rows = STORE.query(
            "SELECT * FROM reflection_patches WHERE tenant_id = ? AND run_id = ? ORDER BY created_at DESC LIMIT ?",
            (tenant.tenant_id, run_id, min(200, max(1, limit))),
        )
    else:
        rows = STORE.query(
            "SELECT * FROM reflection_patches WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
            (tenant.tenant_id, min(200, max(1, limit))),
        )
    return {
        "reflections": [
            {
                "reflection_id": r["reflection_id"],
                "run_id": r["run_id"],
                "verdict": r["verdict"],
                "failure_points": jload(r["failure_points"], []),
                "pivot_actions": jload(r["pivot_actions"], []),
                "patch_text": r["patch_text"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    }


@app.post("/api/distill/optimize")
async def optimize_policy(payload: OptimizeRequest, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    report = await loop.run_in_executor(
        None,
        DISTILLER.optimize,
        tenant.tenant_id,
        payload.run_id,
        payload.learning_rate,
        payload.epochs,
        payload.limit,
    )
    return report


@app.get("/api/distill/samples")
async def distill_samples(tenant: Tenant = Depends(resolve_tenant), limit: int = 50, run_id: Optional[str] = None) -> Dict[str, Any]:
    if run_id:
        rows = STORE.query(
            "SELECT sample_id, run_id, step, reverse_kl, advantage, created_at, tokens FROM distill_samples WHERE tenant_id = ? AND run_id = ? ORDER BY step ASC LIMIT ?",
            (tenant.tenant_id, run_id, min(500, max(1, limit))),
        )
    else:
        rows = STORE.query(
            "SELECT sample_id, run_id, step, reverse_kl, advantage, created_at, tokens FROM distill_samples WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
            (tenant.tenant_id, min(500, max(1, limit))),
        )
    return {
        "samples": [
            {
                "sample_id": r["sample_id"],
                "run_id": r["run_id"],
                "step": int(r["step"]),
                "reverse_kl": float(r["reverse_kl"]),
                "advantage": float(r["advantage"]),
                "token_count": len(jload(r["tokens"], []) or []),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    }


@app.get("/api/policy/weights")
async def policy_weights(tenant: Tenant = Depends(resolve_tenant), limit: int = 200) -> Dict[str, Any]:
    rows = STORE.query(
        "SELECT feature, weight, updates, updated_at FROM policy_weights WHERE tenant_id = ? ORDER BY ABS(weight) DESC LIMIT ?",
        (tenant.tenant_id, min(2000, max(1, limit))),
    )
    return {"weights": [dict(r) for r in rows]}


@app.get("/api/memory/{run_id}")
async def memory_snapshot(run_id: str, tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    if RUNS.get(run_id, tenant.tenant_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    ckpt = CHECKPOINTS.latest(run_id)
    sigma = ckpt["sigma"] if ckpt else empty_sigma()
    query = WM.routing_query(ProceduralSpec.from_dict(jload(RUNS.get(run_id, tenant.tenant_id)["spec"], {}) or {}), sigma, Observation.initial(0, ""))
    skills = EM.search(tenant.tenant_id, query, 3)
    return {
        "run_id": run_id,
        "working_memory": WM.load(tenant.tenant_id, run_id),
        "state": sigma,
        "routed_skills": [{"skill": s.to_dict(), "relevance": sc} for s, sc in skills],
        "cognition": COGNITION.latest(run_id),
    }


@app.get("/api/traces/failures")
async def failure_traces(tenant: Tenant = Depends(resolve_tenant), limit: int = 60) -> Dict[str, Any]:
    return {"failures": TRACES.failures(tenant.tenant_id, min(500, max(1, limit)))}


@app.get("/api/audit")
async def audit_log(tenant: Tenant = Depends(resolve_tenant), limit: int = 200) -> Dict[str, Any]:
    rows = STORE.query(
        "SELECT audit_id, run_id, actor, action, detail, allowed, created_at FROM audit_log WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
        (tenant.tenant_id, min(2000, max(1, limit))),
    )
    return {"audit": [{**dict(r), "detail": jload(r["detail"], {}), "allowed": bool(r["allowed"])} for r in rows]}


@app.get("/api/tools")
async def list_tools(tenant: Tenant = Depends(resolve_tenant)) -> Dict[str, Any]:
    return {
        "tools": [
            {"name": name, "description": spec.description, "schema": spec.schema, "mutating": spec.mutating, "authorized": name in tenant.allowed_tools}
            for name, spec in TOOLS.tools.items()
        ]
    }


@app.get("/api/workspace/{run_id}")
async def workspace_list(run_id: str, tenant: Tenant = Depends(resolve_tenant), path: str = ".") -> Dict[str, Any]:
    if RUNS.get(run_id, tenant.tenant_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    try:
        return WORKSPACE.list_dir(tenant.tenant_id, run_id, path)
    except SecurityError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@app.get("/api/workspace/{run_id}/file")
async def workspace_file(run_id: str, path: str, tenant: Tenant = Depends(resolve_tenant), start: int = 1, end: Optional[int] = None) -> Dict[str, Any]:
    if RUNS.get(run_id, tenant.tenant_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    try:
        return WORKSPACE.read_file(tenant.tenant_id, run_id, path, start, end)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except SecurityError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@app.post("/api/admin/tenants")
async def create_tenant(payload: TenantRequest, _: bool = Depends(require_admin)) -> Dict[str, Any]:
    tenant = TENANTS.ensure_tenant(payload.tenant_id, payload.name, payload.api_key, payload.token_budget)
    return {"tenant_id": tenant.tenant_id, "name": tenant.name, "token_budget": tenant.token_budget, "allowed_tools": tenant.allowed_tools}


@app.get("/api/admin/stats")
async def admin_stats(_: bool = Depends(require_admin)) -> Dict[str, Any]:
    def scalar(sql: str, params: Iterable[Any] = ()) -> int:
        row = STORE.one(sql, params)
        return int(row[0]) if row else 0

    return {
        "tenants": scalar("SELECT COUNT(*) FROM tenants"),
        "runs": scalar("SELECT COUNT(*) FROM runs"),
        "runs_running": scalar("SELECT COUNT(*) FROM runs WHERE status = ?", (RunStatus.RUNNING.value,)),
        "runs_completed": scalar("SELECT COUNT(*) FROM runs WHERE status = ?", (RunStatus.COMPLETED.value,)),
        "runs_failed": scalar("SELECT COUNT(*) FROM runs WHERE status = ?", (RunStatus.FAILED.value,)),
        "checkpoints": scalar("SELECT COUNT(*) FROM checkpoints"),
        "traces": scalar("SELECT COUNT(*) FROM raw_traces"),
        "skills": scalar("SELECT COUNT(*) FROM skills WHERE status = 'active'"),
        "wiki_pages": scalar("SELECT COUNT(*) FROM wiki_pages"),
        "patches_applied": scalar("SELECT COUNT(*) FROM skill_patches WHERE status = 'applied'"),
        "patches_rejected": scalar("SELECT COUNT(*) FROM skill_patches WHERE status = 'rejected'"),
        "distill_samples": scalar("SELECT COUNT(*) FROM distill_samples"),
        "policy_features": scalar("SELECT COUNT(*) FROM policy_weights"),
        "active_workers": ORCHESTRATOR.active_runs(),
        "model_tokens": {"prompt": MODEL.total_prompt_tokens, "completion": MODEL.total_completion_tokens},
    }


@app.post("/api/admin/recover")
async def admin_recover(_: bool = Depends(require_admin)) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    recovered = await loop.run_in_executor(None, ORCHESTRATOR.recover)
    return {"recovered": recovered}


INDEX_FALLBACK = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Autonomous Agent Runtime</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0d11; color: #e6e8ec; margin: 0; padding: 48px; }
main { max-width: 760px; margin: 0 auto; }
h1 { font-weight: 600; letter-spacing: -0.02em; }
code { background: #171a21; padding: 2px 6px; border-radius: 4px; }
a { color: #7aa2f7; }
ul { line-height: 1.9; }
</style>
</head>
<body>
<main>
<h1>Autonomous Agent Runtime</h1>
<p>The backend is running. Place <code>index.html</code> next to <code>main.py</code> (or set <code>AGENT_STATIC</code>) to serve the frontend from this route.</p>
<ul>
<li><a href="/api/health">/api/health</a></li>
<li><a href="/api/docs">/api/docs</a></li>
<li><code>POST /api/chat</code> streaming SSE chat and agent dispatch</li>
<li><code>POST /api/runs</code> launch a durable autonomous run</li>
<li><code>GET /api/runs/{run_id}/stream</code> live run telemetry</li>
<li><code>WS /api/ws</code> bidirectional control channel</li>
</ul>
</main>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    for candidate in (STATIC_ROOT / "index.html", Path.cwd() / "index.html", Path(__file__).resolve().parent / "index.html"):
        if candidate.exists() and candidate.is_file():
            return HTMLResponse(candidate.read_text(encoding="utf-8"))
    return HTMLResponse(INDEX_FALLBACK)


@app.get("/favicon.ico")
async def favicon() -> PlainTextResponse:
    return PlainTextResponse("", status_code=204)


for _static_dir in ("assets", "static", "public"):
    _candidate = STATIC_ROOT / _static_dir
    if _candidate.exists() and _candidate.is_dir():
        app.mount(f"/{_static_dir}", StaticFiles(directory=str(_candidate)), name=_static_dir)


def main() -> None:
    host = os.environ.get("AGENT_HOST", "0.0.0.0")
    port = int(os.environ.get("AGENT_PORT", "8000"))
    log.info("binding %s:%d", host, port)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
        timeout_keep_alive=120,
        access_log=False,
    )


if __name__ == "__main__":
    main()
