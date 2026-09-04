```zig
const std = @import("std");

const Allocator = std.mem.Allocator;
const EMBED_DIM: usize = 64;
const COG_K: usize = 4;
const COG_H: usize = 16;

const SQLITE_OK: c_int = 0;
const SQLITE_ROW: c_int = 100;
const SQLITE_DONE: c_int = 101;
const SQLITE_OPEN_READWRITE: c_int = 0x00000002;
const SQLITE_OPEN_CREATE: c_int = 0x00000004;
const SQLITE_OPEN_FULLMUTEX: c_int = 0x00010000;
const SQLITE_TRANSIENT: isize = -1;

const sqlite3 = opaque {};
const sqlite3_stmt = opaque {};

extern fn sqlite3_open_v2(filename: [*:0]const u8, ppDb: *?*sqlite3, flags: c_int, zVfs: ?[*:0]const u8) c_int;
extern fn sqlite3_close(db: ?*sqlite3) c_int;
extern fn sqlite3_exec(db: ?*sqlite3, sql: [*:0]const u8, callback: ?*const fn (?*anyopaque, c_int, [*c][*c]u8, [*c][*c]u8) callconv(.C) c_int, arg: ?*anyopaque, errmsg: ?*[*c]u8) c_int;
extern fn sqlite3_prepare_v2(db: ?*sqlite3, zSql: [*]const u8, nByte: c_int, ppStmt: *?*sqlite3_stmt, pzTail: ?*[*c]const u8) c_int;
extern fn sqlite3_step(stmt: ?*sqlite3_stmt) c_int;
extern fn sqlite3_finalize(stmt: ?*sqlite3_stmt) c_int;
extern fn sqlite3_bind_text(stmt: ?*sqlite3_stmt, idx: c_int, val: [*]const u8, n: c_int, destructor: ?*const fn (?*anyopaque) callconv(.C) void) c_int;
extern fn sqlite3_bind_int64(stmt: ?*sqlite3_stmt, idx: c_int, val: i64) c_int;
extern fn sqlite3_bind_int(stmt: ?*sqlite3_stmt, idx: c_int, val: c_int) c_int;
extern fn sqlite3_bind_double(stmt: ?*sqlite3_stmt, idx: c_int, val: f64) c_int;
extern fn sqlite3_bind_null(stmt: ?*sqlite3_stmt, idx: c_int) c_int;
extern fn sqlite3_column_text(stmt: ?*sqlite3_stmt, col: c_int) [*c]const u8;
extern fn sqlite3_column_bytes(stmt: ?*sqlite3_stmt, col: c_int) c_int;
extern fn sqlite3_column_int64(stmt: ?*sqlite3_stmt, col: c_int) i64;
extern fn sqlite3_column_int(stmt: ?*sqlite3_stmt, col: c_int) c_int;
extern fn sqlite3_column_double(stmt: ?*sqlite3_stmt, col: c_int) f64;
extern fn sqlite3_errmsg(db: ?*sqlite3) [*c]const u8;
extern fn sqlite3_last_insert_rowid(db: ?*sqlite3) i64;
extern fn sqlite3_busy_timeout(db: ?*sqlite3, ms: c_int) c_int;
extern fn sqlite3_changes(db: ?*sqlite3) c_int;

const Config = struct {
    host: []u8,
    port: u16,
    database_path: []u8,
    modular_api_key: []u8,
    modular_base_url: []u8,
    model: []u8,
    workspace_root: []u8,
    knowledge_root: []u8,
    default_tenant_id: []u8,
    allowed_http_hosts: []u8,
    max_request_bytes: usize,
    max_state_bytes: usize,
    max_prompt_bytes: usize,
    max_response_bytes: usize,
    max_steps: i64,
    default_token_budget: i64,
    model_max_tokens: i64,

    fn load(allocator: Allocator) !Config {
        const port_text = try envOwnedOr(allocator, "PORT", "8080");
        defer allocator.free(port_text);
        const port = try std.fmt.parseInt(u16, port_text, 10);

        const max_request_text = try envOwnedOr(allocator, "AGENT_MAX_REQUEST_BYTES", "10485760");
        defer allocator.free(max_request_text);
        const max_state_text = try envOwnedOr(allocator, "AGENT_MAX_STATE_BYTES", "65536");
        defer allocator.free(max_state_text);
        const max_prompt_text = try envOwnedOr(allocator, "AGENT_MAX_PROMPT_BYTES", "131072");
        defer allocator.free(max_prompt_text);
        const max_response_text = try envOwnedOr(allocator, "AGENT_MAX_RESPONSE_BYTES", "20971520");
        defer allocator.free(max_response_text);
        const max_steps_text = try envOwnedOr(allocator, "AGENT_MAX_STEPS", "100000");
        defer allocator.free(max_steps_text);
        const token_budget_text = try envOwnedOr(allocator, "AGENT_DEFAULT_TOKEN_BUDGET", "10000000");
        defer allocator.free(token_budget_text);
        const model_max_tokens_text = try envOwnedOr(allocator, "AGENT_MODEL_MAX_TOKENS", "100000");
        defer allocator.free(model_max_tokens_text);

        return Config{
            .host = try envOwnedOr(allocator, "HOST", "0.0.0.0"),
            .port = port,
            .database_path = try envOwnedOr(allocator, "AGENT_DATABASE_PATH", "agent_runtime.sqlite3"),
            .modular_api_key = try std.process.getEnvVarOwned(allocator, "MODULAR_API_KEY"),
            .modular_base_url = try envOwnedOr(allocator, "MODULAR_BASE_URL", "https://api.modular.com/v1"),
            .model = try envOwnedOr(allocator, "MODULAR_MODEL", "zai-org/glm-5.3"),
            .workspace_root = try envOwnedOr(allocator, "AGENT_WORKSPACE", "agent_workspace"),
            .knowledge_root = try envOwnedOr(allocator, "AGENT_KNOWLEDGE_ROOT", "agent_knowledge"),
            .default_tenant_id = try envOwnedOr(allocator, "AGENT_DEFAULT_TENANT", "default"),
            .allowed_http_hosts = try envOwnedOr(allocator, "AGENT_ALLOWED_HTTP_HOSTS", "*"),
            .max_request_bytes = try std.fmt.parseInt(usize, max_request_text, 10),
            .max_state_bytes = try std.fmt.parseInt(usize, max_state_text, 10),
            .max_prompt_bytes = try std.fmt.parseInt(usize, max_prompt_text, 10),
            .max_response_bytes = try std.fmt.parseInt(usize, max_response_text, 10),
            .max_steps = try std.fmt.parseInt(i64, max_steps_text, 10),
            .default_token_budget = try std.fmt.parseInt(i64, token_budget_text, 10),
            .model_max_tokens = try std.fmt.parseInt(i64, model_max_tokens_text, 10),
        };
    }

    fn deinit(self: *Config, allocator: Allocator) void {
        allocator.free(self.host);
        allocator.free(self.database_path);
        allocator.free(self.modular_api_key);
        allocator.free(self.modular_base_url);
        allocator.free(self.model);
        allocator.free(self.workspace_root);
        allocator.free(self.knowledge_root);
        allocator.free(self.default_tenant_id);
        allocator.free(self.allowed_http_hosts);
    }
};

fn envOwnedOr(allocator: Allocator, name: []const u8, default_value: []const u8) ![]u8 {
    return std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => allocator.dupe(u8, default_value),
        else => err,
    };
}

fn now() i64 {
    return std.time.timestamp();
}

fn nowMillis() i64 {
    return std.time.milliTimestamp();
}

fn makeId(allocator: Allocator, prefix: []const u8) ![]u8 {
    var bytes: [16]u8 = undefined;
    std.crypto.random.bytes(&bytes);
    const hex = "0123456789abcdef";
    var out = try allocator.alloc(u8, prefix.len + 1 + 32);
    @memcpy(out[0..prefix.len], prefix);
    out[prefix.len] = '_';
    for (bytes, 0..) |b, i| {
        out[prefix.len + 1 + i * 2] = hex[(b >> 4) & 0x0f];
        out[prefix.len + 1 + i * 2 + 1] = hex[b & 0x0f];
    }
    return out;
}

const RunRecord = struct {
    id: []u8,
    tenant_id: []u8,
    procedure_json: []u8,
    status: []u8,
    step: i64,
    latest_observation_json: []u8,
    token_budget_used: i64,
    token_budget_limit: i64,

    fn deinit(self: *RunRecord, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.tenant_id);
        allocator.free(self.procedure_json);
        allocator.free(self.status);
        allocator.free(self.latest_observation_json);
    }
};

const SkillRecord = struct {
    id: []u8,
    name: []u8,
    description: []u8,
    trigger_json: []u8,
    procedure_json: []u8,
    embedding_json: []u8,
    vector_score: f64,
    sparse_rank: usize,
    rrf_score: f64,

    fn deinit(self: *SkillRecord, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.description);
        allocator.free(self.trigger_json);
        allocator.free(self.procedure_json);
        allocator.free(self.embedding_json);
    }
};

const ActionClaim = struct {
    id: i64,
    run_id: []u8,
    step: i64,
    action_json: []u8,

    fn deinit(self: *ActionClaim, allocator: Allocator) void {
        allocator.free(self.run_id);
        allocator.free(self.action_json);
    }
};

const ModelResult = struct {
    content: []u8,
    total_tokens: i64,

    fn deinit(self: *ModelResult, allocator: Allocator) void {
        allocator.free(self.content);
    }
};

const Envelope = struct {
    envelope_json: []u8,
    patch_json: []u8,
    action_json: []u8,
    terminal: bool,
    confidence: f64,

    fn deinit(self: *Envelope, allocator: Allocator) void {
        allocator.free(self.envelope_json);
        allocator.free(self.patch_json);
        allocator.free(self.action_json);
    }
};

const HttpRequest = struct {
    method: []const u8,
    target: []const u8,
    path: []const u8,
    query: []const u8,
    headers: std.StringHashMap([]const u8),
    body: []const u8,
};

const Database = struct {
    allocator: Allocator,
    handle: ?*sqlite3,
    mutex: std.Thread.Mutex,

    fn open(allocator: Allocator, path: []const u8) !Database {
        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);
        var handle: ?*sqlite3 = null;
        const flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX;
        const rc = sqlite3_open_v2(path_z.ptr, &handle, flags, null);
        if (rc != SQLITE_OK) return error.SqliteOpenFailed;
        _ = sqlite3_busy_timeout(handle, 10000);
        return Database{ .allocator = allocator, .handle = handle, .mutex = .{} };
    }

    fn close(self: *Database) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.handle != null) {
            _ = sqlite3_close(self.handle);
            self.handle = null;
        }
    }

    fn check(self: *Database, rc: c_int) !void {
        _ = self;
        if (rc == SQLITE_OK) return;
        return error.SqliteError;
    }

    fn checkStep(self: *Database, rc: c_int) !void {
        _ = self;
        if (rc == SQLITE_ROW or rc == SQLITE_DONE) return;
        return error.SqliteError;
    }

    fn execUnlocked(self: *Database, sql: []const u8) !void {
        const sql_z = try self.allocator.dupeZ(u8, sql);
        defer self.allocator.free(sql_z);
        try self.check(sqlite3_exec(self.handle, sql_z.ptr, null, null, null));
    }

    fn exec(self: *Database, sql: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked(sql);
    }

    fn prepareUnlocked(self: *Database, sql: []const u8) !?*sqlite3_stmt {
        var stmt: ?*sqlite3_stmt = null;
        try self.check(sqlite3_prepare_v2(self.handle, sql.ptr, @intCast(sql.len), &stmt, null));
        return stmt;
    }

    fn finalize(self: *Database, stmt: ?*sqlite3_stmt) void {
        _ = self;
        _ = sqlite3_finalize(stmt);
    }

    fn bindText(self: *Database, stmt: ?*sqlite3_stmt, index: c_int, value: []const u8) !void {
        try self.check(sqlite3_bind_text(stmt, index, value.ptr, @intCast(value.len), @ptrFromInt(@as(usize, @bitCast(SQLITE_TRANSIENT)))));
    }

    fn bindInt64(self: *Database, stmt: ?*sqlite3_stmt, index: c_int, value: i64) !void {
        try self.check(sqlite3_bind_int64(stmt, index, value));
    }

    fn bindInt(self: *Database, stmt: ?*sqlite3_stmt, index: c_int, value: c_int) !void {
        try self.check(sqlite3_bind_int(stmt, index, value));
    }

    fn bindDouble(self: *Database, stmt: ?*sqlite3_stmt, index: c_int, value: f64) !void {
        try self.check(sqlite3_bind_double(stmt, index, value));
    }

    fn bindNull(self: *Database, stmt: ?*sqlite3_stmt, index: c_int) !void {
        try self.check(sqlite3_bind_null(stmt, index));
    }

    fn columnText(self: *Database, stmt: ?*sqlite3_stmt, index: c_int) ![]u8 {
        const ptr = sqlite3_column_text(stmt, index);
        if (ptr == null) return self.allocator.dupe(u8, "");
        const len_i = sqlite3_column_bytes(stmt, index);
        if (len_i < 0) return error.SqliteError;
        const len: usize = @intCast(len_i);
        return self.allocator.dupe(u8, ptr[0..len]);
    }

    fn initSchema(self: *Database) !void {
        try self.exec(
            \\PRAGMA journal_mode=WAL;
            \\PRAGMA synchronous=NORMAL;
            \\PRAGMA foreign_keys=ON;
            \\CREATE TABLE IF NOT EXISTS tenants(id TEXT PRIMARY KEY,name TEXT NOT NULL,created_at INTEGER NOT NULL);
            \\CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,procedure_json TEXT NOT NULL,status TEXT NOT NULL,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL,step INTEGER NOT NULL DEFAULT 0,latest_observation_json TEXT NOT NULL DEFAULT '{"type":"start"}',terminal_result_json TEXT,error TEXT,token_budget_used INTEGER NOT NULL DEFAULT 0,token_budget_limit INTEGER NOT NULL DEFAULT 10000000,finalized INTEGER NOT NULL DEFAULT 0,FOREIGN KEY(tenant_id) REFERENCES tenants(id));
            \\CREATE INDEX IF NOT EXISTS runs_tenant_status_idx ON runs(tenant_id,status);
            \\CREATE TABLE IF NOT EXISTS run_state(run_id TEXT NOT NULL,key TEXT NOT NULL,value_json TEXT NOT NULL,updated_at INTEGER NOT NULL,PRIMARY KEY(run_id,key),FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
            \\CREATE TABLE IF NOT EXISTS checkpoints(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,step INTEGER NOT NULL,state_json TEXT NOT NULL,patch_json TEXT NOT NULL,observation_json TEXT NOT NULL,action_json TEXT NOT NULL,created_at INTEGER NOT NULL,FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
            \\CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant_id TEXT NOT NULL,run_id TEXT,step INTEGER NOT NULL,type TEXT NOT NULL,payload_json TEXT NOT NULL,created_at INTEGER NOT NULL);
            \\CREATE INDEX IF NOT EXISTS events_run_id_idx ON events(run_id,id);
            \\CREATE TABLE IF NOT EXISTS observation_queue(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,observation_json TEXT NOT NULL,consumed INTEGER NOT NULL DEFAULT 0,created_at INTEGER NOT NULL,FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
            \\CREATE INDEX IF NOT EXISTS observation_queue_run_idx ON observation_queue(run_id,consumed,id);
            \\CREATE TABLE IF NOT EXISTS pending_actions(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,step INTEGER NOT NULL,action_json TEXT NOT NULL,status TEXT NOT NULL,result_json TEXT,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL,FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
            \\CREATE INDEX IF NOT EXISTS pending_actions_run_idx ON pending_actions(run_id,status,id);
            \\CREATE TABLE IF NOT EXISTS cognition_frames(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,step INTEGER NOT NULL,frame_json TEXT NOT NULL,generated_at INTEGER NOT NULL,FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
            \\CREATE INDEX IF NOT EXISTS cognition_frames_run_idx ON cognition_frames(run_id,id);
            \\CREATE TABLE IF NOT EXISTS skills(id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,name TEXT NOT NULL,description TEXT NOT NULL,trigger_json TEXT NOT NULL,procedure_json TEXT NOT NULL,embedding_json TEXT NOT NULL,version INTEGER NOT NULL,enabled INTEGER NOT NULL,success_count INTEGER NOT NULL DEFAULT 0,failure_count INTEGER NOT NULL DEFAULT 0,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL,FOREIGN KEY(tenant_id) REFERENCES tenants(id));
            \\CREATE INDEX IF NOT EXISTS skills_tenant_enabled_idx ON skills(tenant_id,enabled);
            \\CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(id UNINDEXED,name,description,procedure);
            \\CREATE TABLE IF NOT EXISTS raw_traces(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant_id TEXT NOT NULL,run_id TEXT NOT NULL,step INTEGER NOT NULL,initial_state_json TEXT NOT NULL,selected_skill_json TEXT NOT NULL,observation_json TEXT NOT NULL,model_envelope_json TEXT NOT NULL,state_patch_json TEXT NOT NULL,action_json TEXT NOT NULL,outcome_json TEXT NOT NULL,post_state_json TEXT NOT NULL,verifier_result_json TEXT NOT NULL,created_at INTEGER NOT NULL);
            \\CREATE TABLE IF NOT EXISTS knowledge_versions(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant_id TEXT NOT NULL,content_markdown TEXT NOT NULL,git_commit TEXT NOT NULL,diff_text TEXT NOT NULL,created_at INTEGER NOT NULL);
            \\CREATE TABLE IF NOT EXISTS skill_patches(id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,source_trace_id INTEGER,status TEXT NOT NULL,patch_json TEXT NOT NULL,validation_report_json TEXT NOT NULL,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL);
            \\CREATE INDEX IF NOT EXISTS skill_patches_status_idx ON skill_patches(tenant_id,status);
            \\CREATE TABLE IF NOT EXISTS regression_tasks(id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,query_text TEXT NOT NULL,expected_skill_id TEXT NOT NULL,created_at INTEGER NOT NULL);
            \\CREATE TABLE IF NOT EXISTS distillation_examples(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant_id TEXT NOT NULL,run_id TEXT NOT NULL,reflection_patch_json TEXT NOT NULL,clean_prompt_json TEXT NOT NULL,teacher_logprobs_json TEXT NOT NULL,student_logprobs_json TEXT NOT NULL,reverse_kl REAL NOT NULL,created_at INTEGER NOT NULL);
        );
    }

    fn ensureTenant(self: *Database, tenant_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT OR IGNORE INTO tenants(id,name,created_at) VALUES(?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        try self.bindText(stmt, 2, tenant_id);
        try self.bindInt64(stmt, 3, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn createRun(self: *Database, tenant_id: []const u8, run_id: []const u8, procedure_json: []const u8, initial_observation_json: []const u8, token_budget_limit: i64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked("BEGIN IMMEDIATE;");
        var committed = false;
        defer if (!committed) self.execUnlocked("ROLLBACK;") catch {};
        const stmt = try self.prepareUnlocked("INSERT INTO runs(id,tenant_id,procedure_json,status,created_at,updated_at,step,latest_observation_json,token_budget_limit) VALUES(?,?,?,?,?,?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        try self.bindText(stmt, 2, tenant_id);
        try self.bindText(stmt, 3, procedure_json);
        try self.bindText(stmt, 4, "running");
        try self.bindInt64(stmt, 5, now());
        try self.bindInt64(stmt, 6, now());
        try self.bindInt64(stmt, 7, 0);
        try self.bindText(stmt, 8, initial_observation_json);
        try self.bindInt64(stmt, 9, token_budget_limit);
        try self.checkStep(sqlite3_step(stmt));

        const initial_state = [_]struct { key: []const u8, value: []const u8 }{
            .{ .key = "goal", .value = "\"\"" },
            .{ .key = "status", .value = "\"planning\"" },
            .{ .key = "progress", .value = "0" },
            .{ .key = "subgoals", .value = "[]" },
            .{ .key = "completed", .value = "[]" },
            .{ .key = "blockers", .value = "[]" },
            .{ .key = "facts", .value = "{}" },
            .{ .key = "constraints", .value = "[]" },
            .{ .key = "artifacts", .value = "{}" },
            .{ .key = "verified_progress", .value = "[]" },
            .{ .key = "unresolved_dependencies", .value = "[]" },
            .{ .key = "environment_constraints", .value = "[]" },
            .{ .key = "active_skill_ids", .value = "[]" },
            .{ .key = "completion", .value = "{\"done\":false,\"reason\":\"\"}" },
            .{ .key = "step_status", .value = "\"initialized\"" },
        };
        for (initial_state) |entry| {
            const istmt = try self.prepareUnlocked("INSERT INTO run_state(run_id,key,value_json,updated_at) VALUES(?,?,?,?);");
            defer self.finalize(istmt);
            try self.bindText(istmt, 1, run_id);
            try self.bindText(istmt, 2, entry.key);
            try self.bindText(istmt, 3, entry.value);
            try self.bindInt64(istmt, 4, now());
            try self.checkStep(sqlite3_step(istmt));
        }
        const state_json = try self.getStateJsonUnlocked(run_id);
        defer self.allocator.free(state_json);
        const cstmt = try self.prepareUnlocked("INSERT INTO checkpoints(run_id,step,state_json,patch_json,observation_json,action_json,created_at) VALUES(?,?,?,?,?,?,?);");
        defer self.finalize(cstmt);
        try self.bindText(cstmt, 1, run_id);
        try self.bindInt64(cstmt, 2, 0);
        try self.bindText(cstmt, 3, state_json);
        try self.bindText(cstmt, 4, "{}");
        try self.bindText(cstmt, 5, initial_observation_json);
        try self.bindText(cstmt, 6, "{\"type\":\"none\"}");
        try self.bindInt64(cstmt, 7, now());
        try self.checkStep(sqlite3_step(cstmt));
        try self.execUnlocked("COMMIT;");
        committed = true;
    }

    fn getRun(self: *Database, run_id: []const u8) !RunRecord {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("SELECT id,tenant_id,procedure_json,status,step,latest_observation_json,token_budget_used,token_budget_limit FROM runs WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc != SQLITE_ROW) return error.NotFound;
        return RunRecord{
            .id = try self.columnText(stmt, 0),
            .tenant_id = try self.columnText(stmt, 1),
            .procedure_json = try self.columnText(stmt, 2),
            .status = try self.columnText(stmt, 3),
            .step = sqlite3_column_int64(stmt, 4),
            .latest_observation_json = try self.columnText(stmt, 5),
            .token_budget_used = sqlite3_column_int64(stmt, 6),
            .token_budget_limit = sqlite3_column_int64(stmt, 7),
        };
    }

    fn getRunStatus(self: *Database, run_id: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("SELECT status FROM runs WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc != SQLITE_ROW) return error.NotFound;
        return self.columnText(stmt, 0);
    }

    fn markRunStatus(self: *Database, run_id: []const u8, status: []const u8, error_json: ?[]const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE runs SET status=?,error=?,updated_at=? WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, status);
        if (error_json) |e| try self.bindText(stmt, 2, e) else try self.bindNull(stmt, 2);
        try self.bindInt64(stmt, 3, now());
        try self.bindText(stmt, 4, run_id);
        try self.checkStep(sqlite3_step(stmt));
    }

    fn markCompleted(self: *Database, run_id: []const u8, result_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE runs SET status='completed',terminal_result_json=?,updated_at=? WHERE id=? AND status IN ('running','queued');");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, result_json);
        try self.bindInt64(stmt, 2, now());
        try self.bindText(stmt, 3, run_id);
        try self.checkStep(sqlite3_step(stmt));
    }

    fn claimFinalization(self: *Database, run_id: []const u8) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE runs SET finalized=1,updated_at=? WHERE id=? AND finalized=0 AND status IN ('completed','failed','stopped');");
        defer self.finalize(stmt);
        try self.bindInt64(stmt, 1, now());
        try self.bindText(stmt, 2, run_id);
        try self.checkStep(sqlite3_step(stmt));
        return sqlite3_changes(self.handle) > 0;
    }

    fn listActiveRunIds(self: *Database) ![][]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var list = std.ArrayList([]u8).init(self.allocator);
        const stmt = try self.prepareUnlocked("SELECT id FROM runs WHERE status IN ('queued','running');");
        defer self.finalize(stmt);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            try list.append(try self.columnText(stmt, 0));
        }
        return list.toOwnedSlice();
    }

    fn listRunsJson(self: *Database, tenant_id: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var out = std.ArrayList(u8).init(self.allocator);
        const w = out.writer();
        try w.writeAll("[");
        var first = true;
        const stmt = try self.prepareUnlocked("SELECT id,status,step,created_at,updated_at,terminal_result_json,error,token_budget_used,token_budget_limit FROM runs WHERE tenant_id=? ORDER BY created_at DESC LIMIT 200;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            if (!first) try w.writeAll(",");
            first = false;
            try w.writeAll("{\"id\":");
            const id = try self.columnText(stmt, 0);
            defer self.allocator.free(id);
            try writeJsonString(w, id);
            try w.writeAll(",\"status\":");
            const status = try self.columnText(stmt, 1);
            defer self.allocator.free(status);
            try writeJsonString(w, status);
            try w.print(",\"step\":{},\"created_at\":{},\"updated_at\":{},\"terminal_result\":", .{ sqlite3_column_int64(stmt, 2), sqlite3_column_int64(stmt, 3), sqlite3_column_int64(stmt, 4) });
            const terminal = try self.columnText(stmt, 5);
            defer self.allocator.free(terminal);
            if (terminal.len == 0) try w.writeAll("null") else try w.writeAll(terminal);
            try w.writeAll(",\"error\":");
            const err = try self.columnText(stmt, 6);
            defer self.allocator.free(err);
            if (err.len == 0) try w.writeAll("null") else try w.writeAll(err);
            try w.print(",\"token_budget_used\":{},\"token_budget_limit\":{}}}", .{ sqlite3_column_int64(stmt, 7), sqlite3_column_int64(stmt, 8) });
        }
        try w.writeAll("]");
        return out.toOwnedSlice();
    }

    fn runJson(self: *Database, tenant_id: []const u8, run_id: []const u8) ![]u8 {
        var run = try self.getRun(run_id);
        defer run.deinit(self.allocator);
        if (!std.mem.eql(u8, run.tenant_id, tenant_id)) return error.NotFound;
        const state = try self.getStateJson(run_id);
        defer self.allocator.free(state);
        var out = std.ArrayList(u8).init(self.allocator);
        const w = out.writer();
        try w.writeAll("{\"id\":");
        try writeJsonString(w, run.id);
        try w.writeAll(",\"tenant_id\":");
        try writeJsonString(w, run.tenant_id);
        try w.writeAll(",\"status\":");
        try writeJsonString(w, run.status);
        try w.print(",\"step\":{},\"procedure\":", .{run.step});
        try w.writeAll(run.procedure_json);
        try w.writeAll(",\"state\":");
        try w.writeAll(state);
        try w.writeAll(",\"latest_observation\":");
        try w.writeAll(run.latest_observation_json);
        try w.print(",\"token_budget_used\":{},\"token_budget_limit\":{}}}", .{ run.token_budget_used, run.token_budget_limit });
        return out.toOwnedSlice();
    }

    fn getStateJson(self: *Database, run_id: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.getStateJsonUnlocked(run_id);
    }

    fn getStateJsonUnlocked(self: *Database, run_id: []const u8) ![]u8 {
        var out = std.ArrayList(u8).init(self.allocator);
        const w = out.writer();
        try w.writeAll("{");
        var first = true;
        const stmt = try self.prepareUnlocked("SELECT key,value_json FROM run_state WHERE run_id=? ORDER BY key ASC;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            if (!first) try w.writeAll(",");
            first = false;
            const key = try self.columnText(stmt, 0);
            defer self.allocator.free(key);
            const value = try self.columnText(stmt, 1);
            defer self.allocator.free(value);
            try writeJsonString(w, key);
            try w.writeAll(":");
            try w.writeAll(value);
        }
        try w.writeAll("}");
        return out.toOwnedSlice();
    }

    fn applyPatchAndCheckpoint(self: *Database, run_id: []const u8, next_step: i64, patch_json: []const u8, observation_json: []const u8, action_json: []const u8, max_state_bytes: usize) ![]u8 {
        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, patch_json, .{});
        defer parsed.deinit();
        try validatePatchValue(parsed.value);
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked("BEGIN IMMEDIATE;");
        var committed = false;
        defer if (!committed) self.execUnlocked("ROLLBACK;") catch {};
        switch (parsed.value) {
            .object => |obj| {
                var it = obj.iterator();
                while (it.next()) |entry| {
                    const key = entry.key_ptr.*;
                    if (containsBannedKey(key)) return error.InvalidStatePatch;
                    switch (entry.value_ptr.*) {
                        .null => {
                            const stmt = try self.prepareUnlocked("DELETE FROM run_state WHERE run_id=? AND key=?;");
                            defer self.finalize(stmt);
                            try self.bindText(stmt, 1, run_id);
                            try self.bindText(stmt, 2, key);
                            try self.checkStep(sqlite3_step(stmt));
                        },
                        else => {
                            const value_json = try jsonValueToOwned(self.allocator, entry.value_ptr.*);
                            defer self.allocator.free(value_json);
                            const stmt = try self.prepareUnlocked("INSERT INTO run_state(run_id,key,value_json,updated_at) VALUES(?,?,?,?) ON CONFLICT(run_id,key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at;");
                            defer self.finalize(stmt);
                            try self.bindText(stmt, 1, run_id);
                            try self.bindText(stmt, 2, key);
                            try self.bindText(stmt, 3, value_json);
                            try self.bindInt64(stmt, 4, now());
                            try self.checkStep(sqlite3_step(stmt));
                        },
                    }
                }
            },
            else => return error.InvalidStatePatch,
        }
        const state_json = try self.getStateJsonUnlocked(run_id);
        if (state_json.len > max_state_bytes) {
            self.allocator.free(state_json);
            return error.StateTooLarge;
        }
        const ustmt = try self.prepareUnlocked("UPDATE runs SET step=?,updated_at=? WHERE id=?;");
        defer self.finalize(ustmt);
        try self.bindInt64(ustmt, 1, next_step);
        try self.bindInt64(ustmt, 2, now());
        try self.bindText(ustmt, 3, run_id);
        try self.checkStep(sqlite3_step(ustmt));
        const cstmt = try self.prepareUnlocked("INSERT INTO checkpoints(run_id,step,state_json,patch_json,observation_json,action_json,created_at) VALUES(?,?,?,?,?,?,?);");
        defer self.finalize(cstmt);
        try self.bindText(cstmt, 1, run_id);
        try self.bindInt64(cstmt, 2, next_step);
        try self.bindText(cstmt, 3, state_json);
        try self.bindText(cstmt, 4, patch_json);
        try self.bindText(cstmt, 5, observation_json);
        try self.bindText(cstmt, 6, action_json);
        try self.bindInt64(cstmt, 7, now());
        try self.checkStep(sqlite3_step(cstmt));
        try self.execUnlocked("COMMIT;");
        committed = true;
        return state_json;
    }

    fn insertEvent(self: *Database, tenant_id: []const u8, run_id: ?[]const u8, step: i64, typ: []const u8, payload_json: []const u8) !i64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO events(tenant_id,run_id,step,type,payload_json,created_at) VALUES(?,?,?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        if (run_id) |rid| try self.bindText(stmt, 2, rid) else try self.bindNull(stmt, 2);
        try self.bindInt64(stmt, 3, step);
        try self.bindText(stmt, 4, typ);
        try self.bindText(stmt, 5, payload_json);
        try self.bindInt64(stmt, 6, now());
        try self.checkStep(sqlite3_step(stmt));
        return sqlite3_last_insert_rowid(self.handle);
    }

    fn enqueueObservation(self: *Database, run_id: []const u8, observation_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO observation_queue(run_id,observation_json,created_at) VALUES(?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        try self.bindText(stmt, 2, observation_json);
        try self.bindInt64(stmt, 3, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn takeObservation(self: *Database, run_id: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked("BEGIN IMMEDIATE;");
        var committed = false;
        defer if (!committed) self.execUnlocked("ROLLBACK;") catch {};
        const stmt = try self.prepareUnlocked("SELECT id,observation_json FROM observation_queue WHERE run_id=? AND consumed=0 ORDER BY id ASC LIMIT 1;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc == SQLITE_ROW) {
            const obs_id = sqlite3_column_int64(stmt, 0);
            const obs = try self.columnText(stmt, 1);
            const ustmt = try self.prepareUnlocked("UPDATE observation_queue SET consumed=1 WHERE id=?;");
            defer self.finalize(ustmt);
            try self.bindInt64(ustmt, 1, obs_id);
            try self.checkStep(sqlite3_step(ustmt));
            const rstmt = try self.prepareUnlocked("UPDATE runs SET latest_observation_json=?,updated_at=? WHERE id=?;");
            defer self.finalize(rstmt);
            try self.bindText(rstmt, 1, obs);
            try self.bindInt64(rstmt, 2, now());
            try self.bindText(rstmt, 3, run_id);
            try self.checkStep(sqlite3_step(rstmt));
            try self.execUnlocked("COMMIT;");
            committed = true;
            return obs;
        }
        const rstmt = try self.prepareUnlocked("SELECT latest_observation_json FROM runs WHERE id=?;");
        defer self.finalize(rstmt);
        try self.bindText(rstmt, 1, run_id);
        const rrc = sqlite3_step(rstmt);
        try self.checkStep(rrc);
        if (rrc != SQLITE_ROW) return error.NotFound;
        const obs = try self.columnText(rstmt, 0);
        try self.execUnlocked("COMMIT;");
        committed = true;
        return obs;
    }

    fn updateLatestObservation(self: *Database, run_id: []const u8, observation_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE runs SET latest_observation_json=?,updated_at=? WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, observation_json);
        try self.bindInt64(stmt, 2, now());
        try self.bindText(stmt, 3, run_id);
        try self.checkStep(sqlite3_step(stmt));
    }

    fn enqueueAction(self: *Database, run_id: []const u8, step: i64, action_json: []const u8) !i64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO pending_actions(run_id,step,action_json,status,created_at,updated_at) VALUES(?,?,?,'pending',?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        try self.bindInt64(stmt, 2, step);
        try self.bindText(stmt, 3, action_json);
        try self.bindInt64(stmt, 4, now());
        try self.bindInt64(stmt, 5, now());
        try self.checkStep(sqlite3_step(stmt));
        return sqlite3_last_insert_rowid(self.handle);
    }

    fn hasOpenAction(self: *Database, run_id: []const u8) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("SELECT id FROM pending_actions WHERE run_id=? AND status IN ('pending','executing') LIMIT 1;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        return rc == SQLITE_ROW;
    }

    fn claimPendingAction(self: *Database, run_id: []const u8) !?ActionClaim {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked("BEGIN IMMEDIATE;");
        var committed = false;
        defer if (!committed) self.execUnlocked("ROLLBACK;") catch {};
        const stmt = try self.prepareUnlocked("SELECT id,run_id,step,action_json FROM pending_actions WHERE run_id=? AND status='pending' ORDER BY id ASC LIMIT 1;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc != SQLITE_ROW) {
            try self.execUnlocked("COMMIT;");
            committed = true;
            return null;
        }
        const id = sqlite3_column_int64(stmt, 0);
        const rid = try self.columnText(stmt, 1);
        const step = sqlite3_column_int64(stmt, 2);
        const action_json = try self.columnText(stmt, 3);
        const ustmt = try self.prepareUnlocked("UPDATE pending_actions SET status='executing',updated_at=? WHERE id=? AND status='pending';");
        defer self.finalize(ustmt);
        try self.bindInt64(ustmt, 1, now());
        try self.bindInt64(ustmt, 2, id);
        try self.checkStep(sqlite3_step(ustmt));
        try self.execUnlocked("COMMIT;");
        committed = true;
        return ActionClaim{ .id = id, .run_id = rid, .step = step, .action_json = action_json };
    }

    fn completeAction(self: *Database, action_id: i64, outcome_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE pending_actions SET status='done',result_json=?,updated_at=? WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, outcome_json);
        try self.bindInt64(stmt, 2, now());
        try self.bindInt64(stmt, 3, action_id);
        try self.checkStep(sqlite3_step(stmt));
    }

    fn getActionResult(self: *Database, action_id: i64) !?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("SELECT result_json FROM pending_actions WHERE id=? AND status='done';");
        defer self.finalize(stmt);
        try self.bindInt64(stmt, 1, action_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc != SQLITE_ROW) return null;
        return self.columnText(stmt, 0);
    }

    fn insertCognitionFrame(self: *Database, run_id: []const u8, step: i64, frame_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO cognition_frames(run_id,step,frame_json,generated_at) VALUES(?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        try self.bindInt64(stmt, 2, step);
        try self.bindText(stmt, 3, frame_json);
        try self.bindInt64(stmt, 4, nowMillis());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn latestCognitionGeneratedAt(self: *Database, run_id: []const u8) !i64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("SELECT generated_at FROM cognition_frames WHERE run_id=? ORDER BY id DESC LIMIT 1;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc != SQLITE_ROW) return nowMillis();
        return sqlite3_column_int64(stmt, 0);
    }

    fn addRawTrace(self: *Database, tenant_id: []const u8, run_id: []const u8, step: i64, initial_state_json: []const u8, selected_skill_json: []const u8, observation_json: []const u8, model_envelope_json: []const u8, state_patch_json: []const u8, action_json: []const u8, outcome_json: []const u8, post_state_json: []const u8, verifier_result_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO raw_traces(tenant_id,run_id,step,initial_state_json,selected_skill_json,observation_json,model_envelope_json,state_patch_json,action_json,outcome_json,post_state_json,verifier_result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        try self.bindText(stmt, 2, run_id);
        try self.bindInt64(stmt, 3, step);
        try self.bindText(stmt, 4, initial_state_json);
        try self.bindText(stmt, 5, selected_skill_json);
        try self.bindText(stmt, 6, observation_json);
        try self.bindText(stmt, 7, model_envelope_json);
        try self.bindText(stmt, 8, state_patch_json);
        try self.bindText(stmt, 9, action_json);
        try self.bindText(stmt, 10, outcome_json);
        try self.bindText(stmt, 11, post_state_json);
        try self.bindText(stmt, 12, verifier_result_json);
        try self.bindInt64(stmt, 13, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn loadEnabledSkills(self: *Database, tenant_id: []const u8) ![]SkillRecord {
        self.mutex.lock();
        defer self.mutex.unlock();
        var list = std.ArrayList(SkillRecord).init(self.allocator);
        const stmt = try self.prepareUnlocked("SELECT id,name,description,trigger_json,procedure_json,embedding_json FROM skills WHERE tenant_id=? AND enabled=1 ORDER BY updated_at DESC LIMIT 1000;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            try list.append(SkillRecord{
                .id = try self.columnText(stmt, 0),
                .name = try self.columnText(stmt, 1),
                .description = try self.columnText(stmt, 2),
                .trigger_json = try self.columnText(stmt, 3),
                .procedure_json = try self.columnText(stmt, 4),
                .embedding_json = try self.columnText(stmt, 5),
                .vector_score = 0,
                .sparse_rank = std.math.maxInt(usize),
                .rrf_score = 0,
            });
        }
        return list.toOwnedSlice();
    }

    fn searchSkillFtsIds(self: *Database, tenant_id: []const u8, fts_query: []const u8) ![][]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var ids = std.ArrayList([]u8).init(self.allocator);
        const stmt = try self.prepareUnlocked("SELECT skills.id FROM skill_fts JOIN skills ON skills.id=skill_fts.id WHERE skill_fts MATCH ? AND skills.tenant_id=? AND skills.enabled=1 ORDER BY bm25(skill_fts) LIMIT 20;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, fts_query);
        try self.bindText(stmt, 2, tenant_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            try ids.append(try self.columnText(stmt, 0));
        }
        return ids.toOwnedSlice();
    }

    fn addSkillFromFields(self: *Database, tenant_id: []const u8, id: []const u8, name: []const u8, description: []const u8, trigger_json: []const u8, procedure_json: []const u8, enabled: bool) !void {
        var vector: [EMBED_DIM]f64 = undefined;
        var combined = std.ArrayList(u8).init(self.allocator);
        defer combined.deinit();
        try combined.writer().print("{s}\n{s}\n{s}", .{ name, description, procedure_json });
        embedText(combined.items, &vector);
        const embedding_json = try embeddingToJson(self.allocator, &vector);
        defer self.allocator.free(embedding_json);
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked("BEGIN IMMEDIATE;");
        var committed = false;
        defer if (!committed) self.execUnlocked("ROLLBACK;") catch {};
        const stmt = try self.prepareUnlocked("INSERT INTO skills(id,tenant_id,name,description,trigger_json,procedure_json,embedding_json,version,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,description=excluded.description,trigger_json=excluded.trigger_json,procedure_json=excluded.procedure_json,embedding_json=excluded.embedding_json,version=skills.version+1,enabled=excluded.enabled,updated_at=excluded.updated_at;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, id);
        try self.bindText(stmt, 2, tenant_id);
        try self.bindText(stmt, 3, name);
        try self.bindText(stmt, 4, description);
        try self.bindText(stmt, 5, trigger_json);
        try self.bindText(stmt, 6, procedure_json);
        try self.bindText(stmt, 7, embedding_json);
        try self.bindInt64(stmt, 8, 1);
        try self.bindInt(stmt, 9, if (enabled) 1 else 0);
        try self.bindInt64(stmt, 10, now());
        try self.bindInt64(stmt, 11, now());
        try self.checkStep(sqlite3_step(stmt));
        const dstmt = try self.prepareUnlocked("DELETE FROM skill_fts WHERE id=?;");
        defer self.finalize(dstmt);
        try self.bindText(dstmt, 1, id);
        try self.checkStep(sqlite3_step(dstmt));
        const fstmt = try self.prepareUnlocked("INSERT INTO skill_fts(id,name,description,procedure) VALUES(?,?,?,?);");
        defer self.finalize(fstmt);
        try self.bindText(fstmt, 1, id);
        try self.bindText(fstmt, 2, name);
        try self.bindText(fstmt, 3, description);
        try self.bindText(fstmt, 4, procedure_json);
        try self.checkStep(sqlite3_step(fstmt));
        try self.execUnlocked("COMMIT;");
        committed = true;
    }

    fn incrementTokenUsage(self: *Database, run_id: []const u8, amount: i64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE runs SET token_budget_used=token_budget_used+?,updated_at=? WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindInt64(stmt, 1, amount);
        try self.bindInt64(stmt, 2, now());
        try self.bindText(stmt, 3, run_id);
        try self.checkStep(sqlite3_step(stmt));
    }

    fn eventsSinceJson(self: *Database, tenant_id: []const u8, run_id: []const u8, last_id: i64) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var out = std.ArrayList(u8).init(self.allocator);
        const w = out.writer();
        try w.writeAll("[");
        var first = true;
        const stmt = try self.prepareUnlocked("SELECT id,step,type,payload_json,created_at FROM events WHERE tenant_id=? AND run_id=? AND id>? ORDER BY id ASC LIMIT 200;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        try self.bindText(stmt, 2, run_id);
        try self.bindInt64(stmt, 3, last_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            if (!first) try w.writeAll(",");
            first = false;
            const typ = try self.columnText(stmt, 2);
            defer self.allocator.free(typ);
            const payload = try self.columnText(stmt, 3);
            defer self.allocator.free(payload);
            try w.print("{{\"id\":{},\"step\":{},\"type\":", .{ sqlite3_column_int64(stmt, 0), sqlite3_column_int64(stmt, 1) });
            try writeJsonString(w, typ);
            try w.writeAll(",\"payload\":");
            try w.writeAll(if (payload.len == 0) "null" else payload);
            try w.print(",\"created_at\":{}}}", .{sqlite3_column_int64(stmt, 4)});
        }
        try w.writeAll("]");
        return out.toOwnedSlice();
    }

    fn insertSkillPatch(self: *Database, tenant_id: []const u8, patch_id: []const u8, source_trace_id: ?i64, status: []const u8, patch_json: []const u8, validation_report_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO skill_patches(id,tenant_id,source_trace_id,status,patch_json,validation_report_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,validation_report_json=excluded.validation_report_json,updated_at=excluded.updated_at;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, patch_id);
        try self.bindText(stmt, 2, tenant_id);
        if (source_trace_id) |sid| try self.bindInt64(stmt, 3, sid) else try self.bindNull(stmt, 3);
        try self.bindText(stmt, 4, status);
        try self.bindText(stmt, 5, patch_json);
        try self.bindText(stmt, 6, validation_report_json);
        try self.bindInt64(stmt, 7, now());
        try self.bindInt64(stmt, 8, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn insertDistillationExample(self: *Database, tenant_id: []const u8, run_id: []const u8, reflection_patch_json: []const u8, clean_prompt_json: []const u8, teacher_logprobs_json: []const u8, student_logprobs_json: []const u8, reverse_kl: f64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO distillation_examples(tenant_id,run_id,reflection_patch_json,clean_prompt_json,teacher_logprobs_json,student_logprobs_json,reverse_kl,created_at) VALUES(?,?,?,?,?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        try self.bindText(stmt, 2, run_id);
        try self.bindText(stmt, 3, reflection_patch_json);
        try self.bindText(stmt, 4, clean_prompt_json);
        try self.bindText(stmt, 5, teacher_logprobs_json);
        try self.bindText(stmt, 6, student_logprobs_json);
        try self.bindDouble(stmt, 7, reverse_kl);
        try self.bindInt64(stmt, 8, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn buildKnowledgeMarkdown(self: *Database, tenant_id: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var out = std.ArrayList(u8).init(self.allocator);
        const w = out.writer();
        try w.writeAll("# Autonomous Agent Playbook\n\n## Verified Skills\n\n");
        var stmt = try self.prepareUnlocked("SELECT name,description,procedure_json,success_count,failure_count,version FROM skills WHERE tenant_id=? AND enabled=1 ORDER BY updated_at DESC LIMIT 200;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            const name = try self.columnText(stmt, 0);
            defer self.allocator.free(name);
            const description = try self.columnText(stmt, 1);
            defer self.allocator.free(description);
            const procedure = try self.columnText(stmt, 2);
            defer self.allocator.free(procedure);
            try w.print("### {s}\n\n{s}\n\nVersion: {}\nSuccesses: {}\nFailures: {}\n\nProcedure:\n\n```json\n{s}\n```\n\n", .{ name, description, sqlite3_column_int64(stmt, 5), sqlite3_column_int64(stmt, 3), sqlite3_column_int64(stmt, 4), procedure });
        }
        return out.toOwnedSlice();
    }

    fn insertKnowledgeVersion(self: *Database, tenant_id: []const u8, content: []const u8, commit: []const u8, diff: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO knowledge_versions(tenant_id,content_markdown,git_commit,diff_text,created_at) VALUES(?,?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        try self.bindText(stmt, 2, content);
        try self.bindText(stmt, 3, commit);
        try self.bindText(stmt, 4, diff);
        try self.bindInt64(stmt, 5, now());
        try self.checkStep(sqlite3_step(stmt));
    }
};

const App = struct {
    allocator: Allocator,
    config: Config,
    db: *Database,
    fs_mutex: std.Thread.Mutex,
};

const ExprEval = struct {
    src: []const u8,
    pos: usize,

    pub fn skipWs(self: *ExprEval) void {
        while (self.pos < self.src.len and (self.src[self.pos] == ' ' or self.src[self.pos] == '\t' or self.src[self.pos] == '\r' or self.src[self.pos] == '\n')) self.pos += 1;
    }

    pub fn parseExpr(self: *ExprEval) !f64 {
        var left = try self.parseTerm();
        while (true) {
            self.skipWs();
            if (self.pos >= self.src.len) break;
            const ch = self.src[self.pos];
            if (ch == '+') {
                self.pos += 1;
                const right = try self.parseTerm();
                left += right;
            } else if (ch == '-') {
                self.pos += 1;
                const right = try self.parseTerm();
                left -= right;
            } else break;
        }
        return left;
    }

    fn parseTerm(self: *ExprEval) !f64 {
        var left = try self.parseUnary();
        while (true) {
            self.skipWs();
            if (self.pos >= self.src.len) break;
            const ch = self.src[self.pos];
            if (ch == '*') {
                if (self.pos + 1 < self.src.len and self.src[self.pos + 1] == '*') break;
                self.pos += 1;
                const right = try self.parseUnary();
                left *= right;
            } else if (ch == '/') {
                self.pos += 1;
                const right = try self.parseUnary();
                if (right == 0) return error.DivisionByZero;
                left /= right;
            } else if (ch == '%') {
                self.pos += 1;
                const right = try self.parseUnary();
                if (right == 0) return error.DivisionByZero;
                left = @mod(left, right);
            } else break;
        }
        return left;
    }

    fn parseUnary(self: *ExprEval) anyerror!f64 {
        self.skipWs();
        if (self.pos < self.src.len and self.src[self.pos] == '-') {
            self.pos += 1;
            const v = try self.parseUnary();
            return -v;
        }
        if (self.pos < self.src.len and self.src[self.pos] == '+') {
            self.pos += 1;
            return self.parseUnary();
        }
        return self.parsePower();
    }

    fn parsePower(self: *ExprEval) anyerror!f64 {
        const base = try self.parseAtom();
        self.skipWs();
        if (self.pos + 1 < self.src.len and self.src[self.pos] == '*' and self.src[self.pos + 1] == '*') {
            self.pos += 2;
            const exp = try self.parseUnary();
            return std.math.pow(f64, base, exp);
        }
        if (self.pos < self.src.len and self.src[self.pos] == '^') {
            self.pos += 1;
            const exp = try self.parseUnary();
            return std.math.pow(f64, base, exp);
        }
        return base;
    }

    fn parseAtom(self: *ExprEval) anyerror!f64 {
        self.skipWs();
        if (self.pos >= self.src.len) return error.ParseFailure;
        const ch = self.src[self.pos];
        if (ch == '(') {
            self.pos += 1;
            const v = try self.parseExpr();
            self.skipWs();
            if (self.pos >= self.src.len or self.src[self.pos] != ')') return error.ParseFailure;
            self.pos += 1;
            return v;
        }
        if (std.ascii.isAlphabetic(ch)) {
            const start = self.pos;
            while (self.pos < self.src.len and std.ascii.isAlphabetic(self.src[self.pos])) self.pos += 1;
            const name = self.src[start..self.pos];
            self.skipWs();
            if (std.mem.eql(u8, name, "pi")) return std.math.pi;
            if (std.mem.eql(u8, name, "e")) return std.math.e;
            if (self.pos >= self.src.len or self.src[self.pos] != '(') return error.ParseFailure;
            self.pos += 1;
            const arg = try self.parseExpr();
            self.skipWs();
            if (self.pos >= self.src.len or self.src[self.pos] != ')') return error.ParseFailure;
            self.pos += 1;
            if (std.mem.eql(u8, name, "sqrt")) return @sqrt(arg);
            if (std.mem.eql(u8, name, "abs")) return @abs(arg);
            if (std.mem.eql(u8, name, "sin")) return @sin(arg);
            if (std.mem.eql(u8, name, "cos")) return @cos(arg);
            if (std.mem.eql(u8, name, "tan")) return @tan(arg);
            if (std.mem.eql(u8, name, "log")) {
                if (arg <= 0) return error.ParseFailure;
                return std.math.log(f64, std.math.e, arg);
            }
            if (std.mem.eql(u8, name, "exp")) return std.math.exp(arg);
            if (std.mem.eql(u8, name, "floor")) return @floor(arg);
            if (std.mem.eql(u8, name, "ceil")) return @ceil(arg);
            if (std.mem.eql(u8, name, "round")) return @round(arg);
            return error.ParseFailure;
        }
        const start = self.pos;
        var seen_dot = false;
        while (self.pos < self.src.len) {
            const cc = self.src[self.pos];
            if (std.ascii.isDigit(cc)) {
                self.pos += 1;
            } else if (cc == '.' and !seen_dot) {
                seen_dot = true;
                self.pos += 1;
            } else if ((cc == 'e' or cc == 'E') and self.pos > start) {
                if (self.pos + 1 < self.src.len and (std.ascii.isDigit(self.src[self.pos + 1]) or self.src[self.pos + 1] == '-' or self.src[self.pos + 1] == '+')) {
                    self.pos += 2;
                } else break;
            } else break;
        }
        if (self.pos == start) return error.ParseFailure;
        return std.fmt.parseFloat(f64, self.src[start..self.pos]) catch error.ParseFailure;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var config = try Config.load(allocator);
    defer config.deinit(allocator);
    try std.fs.cwd().makePath(config.workspace_root);
    try std.fs.cwd().makePath(config.knowledge_root);
    var db = try Database.open(allocator, config.database_path);
    defer db.close();
    try db.initSchema();
    try db.ensureTenant(config.default_tenant_id);
    var app = App{ .allocator = allocator, .config = config, .db = &db, .fs_mutex = .{} };
    try seedInitialSkills(&app);
    try resumeActiveRuns(&app);
    const meta_thread = try std.Thread.spawn(.{}, metaHarnessEntry, .{&app});
    meta_thread.detach();
    try startHttpServer(&app);
}

fn seedInitialSkills(app: *App) !void {
    const tenant_id = app.config.default_tenant_id;
    try app.db.addSkillFromFields(tenant_id, "skill_decompose_goal", "Goal Decomposition", "When status is planning or subgoals list is empty", "{\"trigger\":\"planning\"}", "{\"procedure\":\"1. Split goal into 3-7 subgoals. 2. Record constraints. 3. Update status to executing.\"}", true);
    try app.db.addSkillFromFields(tenant_id, "skill_append_unique_log", "Append Unique Log Line", "When logging without duplicates", "{\"trigger\":\"log\"}", "{\"procedure\":\"1. Use filesystem.append_file with unique true. 2. Verify lines appended.\"}", true);
    try app.db.addSkillFromFields(tenant_id, "skill_numeric_verification", "Numeric Verification", "When computing closed form arithmetic without hallucination", "{\"trigger\":\"arithmetic\"}", "{\"procedure\":\"1. Formulate mathematical expression. 2. Call compute tool. 3. Record verified result into facts.\"}", true);
}

fn resumeActiveRuns(app: *App) !void {
    const ids = try app.db.listActiveRunIds();
    defer {
        for (ids) |id| app.allocator.free(id);
        app.allocator.free(ids);
    }
    for (ids) |id| try spawnRun(app, id);
}

fn spawnRun(app: *App, run_id: []const u8) !void {
    const r1 = try app.allocator.dupe(u8, run_id);
    const t2 = try std.Thread.spawn(.{}, system2Entry, .{ app, r1 });
    t2.detach();
    const r2 = try app.allocator.dupe(u8, run_id);
    const t1 = try std.Thread.spawn(.{}, system1Entry, .{ app, r2 });
    t1.detach();
}

fn system2Entry(app: *App, run_id: []u8) void {
    defer app.allocator.free(run_id);
    runSystem2(app, run_id) catch |err| {
        const err_json = std.fmt.allocPrint(app.allocator, "{{\"error\":\"{s}\"}}", .{@errorName(err)}) catch "{\"error\":\"system2\"}";
        defer if (err_json.ptr != "{\"error\":\"system2\"}".ptr) app.allocator.free(err_json);
        app.db.markRunStatus(run_id, "failed", err_json) catch {};
    };
    finalizeTrajectory(app, run_id) catch {};
}

fn system1Entry(app: *App, run_id: []u8) void {
    defer app.allocator.free(run_id);
    runSystem1(app, run_id) catch |err| {
        const err_json = std.fmt.allocPrint(app.allocator, "{{\"error\":\"{s}\"}}", .{@errorName(err)}) catch return;
        defer app.allocator.free(err_json);
        app.db.insertEvent(app.config.default_tenant_id, run_id, 0, "system1_error", err_json) catch {};
    };
}

fn runSystem2(app: *App, run_id: []const u8) !void {
    while (true) {
        var run = app.db.getRun(run_id) catch |err| {
            if (err == error.NotFound) return;
            return err;
        };
        defer run.deinit(app.allocator);
        if (!std.mem.eql(u8, run.status, "running") and !std.mem.eql(u8, run.status, "queued")) break;
        if (run.step >= app.config.max_steps) {
            const err_json = try std.fmt.allocPrint(app.allocator, "{{\"error\":\"max_steps_exceeded\",\"max_steps\":{}}}", .{app.config.max_steps});
            defer app.allocator.free(err_json);
            try app.db.markRunStatus(run_id, "failed", err_json);
            break;
        }
        if (try app.db.hasOpenAction(run_id)) {
            std.time.sleep(50 * std.time.ns_per_ms);
            continue;
        }
        const initial_state = try app.db.getStateJson(run_id);
        defer app.allocator.free(initial_state);
        if (initial_state.len > app.config.max_state_bytes) return error.StateTooLarge;
        const observation = try app.db.takeObservation(run_id);
        defer app.allocator.free(observation);
        const skills = try routeSkills(app, run.tenant_id, initial_state, observation);
        defer freeSkillRecords(app.allocator, skills);
        const skills_json = try selectedSkillsJson(app.allocator, skills);
        defer app.allocator.free(skills_json);
        const prompt = try buildStepPrompt(app.allocator, run.procedure_json, initial_state, observation, skills_json);
        defer app.allocator.free(prompt);
        if (prompt.len > app.config.max_prompt_bytes) return error.PromptTooLarge;

        const frame = try buildCognitionFrame(app.allocator, initial_state, observation, run.step);
        defer app.allocator.free(frame);
        try app.db.insertCognitionFrame(run_id, run.step, frame);

        var model_result = try callModelStreaming(app, prompt);
        defer model_result.deinit(app.allocator);
        const token_charge = if (model_result.total_tokens > 0) model_result.total_tokens else estimateTokens(prompt.len + model_result.content.len);
        try app.db.incrementTokenUsage(run_id, token_charge);

        var envelope = validateEnvelope(app, model_result.content) catch |err| {
            var full = std.ArrayList(u8).init(app.allocator);
            defer full.deinit();
            const w = full.writer();
            try w.print("{{\"error\":\"{s}\",\"model_output\":", .{@errorName(err)});
            try writeJsonString(w, model_result.content);
            try w.writeAll("}");
            const err_payload = try full.toOwnedSlice();
            defer app.allocator.free(err_payload);
            try app.db.insertEvent(run.tenant_id, run_id, run.step, "validation_failed", err_payload);
            const obs = try std.fmt.allocPrint(app.allocator, "{{\"type\":\"validation_error\",\"error\":\"{s}\",\"instruction\":\"Return valid JSON envelope with state_patch and action.\"}}", .{@errorName(err)});
            defer app.allocator.free(obs);
            try app.db.updateLatestObservation(run_id, obs);
            std.time.sleep(500 * std.time.ns_per_ms);
            continue;
        };
        defer envelope.deinit(app.allocator);

        const next_step = run.step + 1;
        const post_state = try app.db.applyPatchAndCheckpoint(run_id, next_step, envelope.patch_json, observation, envelope.action_json, app.config.max_state_bytes);
        defer app.allocator.free(post_state);

        const action_id = try app.db.enqueueAction(run_id, next_step, envelope.action_json);
        const outcome = try waitActionOutcome(app, action_id, 600000);
        defer app.allocator.free(outcome);

        const post_action_state = try app.db.getStateJson(run_id);
        defer app.allocator.free(post_action_state);
        const verifier = try evaluateStepVerifier(app.allocator, post_action_state, outcome);
        defer app.allocator.free(verifier);

        try app.db.addRawTrace(run.tenant_id, run_id, next_step, initial_state, skills_json, observation, envelope.envelope_json, envelope.patch_json, envelope.action_json, outcome, post_action_state, verifier);
        try app.db.insertEvent(run.tenant_id, run_id, next_step, "step_completed", outcome);

        const status = try app.db.getRunStatus(run_id);
        defer app.allocator.free(status);
        if (!std.mem.eql(u8, status, "running") and !std.mem.eql(u8, status, "queued")) break;
        std.time.sleep(50 * std.time.ns_per_ms);
    }
}

fn runSystem1(app: *App, run_id: []const u8) !void {
    while (true) {
        const status = try app.db.getRunStatus(run_id);
        defer app.allocator.free(status);
        if (!std.mem.eql(u8, status, "running") and !std.mem.eql(u8, status, "queued")) break;
        if (try app.db.claimPendingAction(run_id)) |claim| {
            var c = claim;
            defer c.deinit(app.allocator);
            const outcome = executeAuthorizedAction(app, c.run_id, c.step, c.action_json) catch |err| blk: {
                break :blk try std.fmt.allocPrint(app.allocator, "{{\"ok\":false,\"type\":\"tool_error\",\"error\":\"{s}\",\"at\":{}}}", .{ @errorName(err), now() });
            };
            defer app.allocator.free(outcome);
            try app.db.completeAction(c.id, outcome);
            try app.db.updateLatestObservation(c.run_id, outcome);
        }
        std.time.sleep(20 * std.time.ns_per_ms);
    }
}

fn waitActionOutcome(app: *App, action_id: i64, timeout_ms: i64) ![]u8 {
    const start = nowMillis();
    while (nowMillis() - start < timeout_ms) {
        if (try app.db.getActionResult(action_id)) |result| return result;
        std.time.sleep(20 * std.time.ns_per_ms);
    }
    return error.ActionTimeout;
}

fn routeSkills(app: *App, tenant_id: []const u8, state_json: []const u8, observation_json: []const u8) ![]SkillRecord {
    var query_builder = std.ArrayList(u8).init(app.allocator);
    defer query_builder.deinit();
    try query_builder.writer().print("{s}\n{s}", .{ state_json, observation_json });
    var query_vector: [EMBED_DIM]f64 = undefined;
    embedText(query_builder.items, &query_vector);
    var skills = try app.db.loadEnabledSkills(tenant_id);
    errdefer freeSkillRecords(app.allocator, skills);
    var fts_query = try makeFtsQuery(app.allocator, query_builder.items);
    defer app.allocator.free(fts_query);
    var fts_ids: [][]u8 = &[_][]u8{};
    var have_fts = false;
    if (fts_query.len > 0) {
        fts_ids = app.db.searchSkillFtsIds(tenant_id, fts_query) catch &[_][]u8{};
        have_fts = fts_ids.len > 0;
    }
    defer if (have_fts) {
        for (fts_ids) |id| app.allocator.free(id);
        app.allocator.free(fts_ids);
    };
    for (skills) |*skill| {
        var vec = parseEmbedding(app.allocator, skill.embedding_json) catch blk: {
            var rebuilt: [EMBED_DIM]f64 = undefined;
            embedText(skill.procedure_json, &rebuilt);
            break :blk rebuilt;
        };
        skill.vector_score = cosine(&query_vector, &vec);
        skill.sparse_rank = findRank(fts_ids, skill.id);
    }
    var selected = std.ArrayList(SkillRecord).init(app.allocator);
    var taken = try app.allocator.alloc(bool, skills.len);
    defer app.allocator.free(taken);
    @memset(taken, false);
    var round: usize = 0;
    while (round < 2 and round < skills.len) : (round += 1) {
        var best_index: ?usize = null;
        var best_score: f64 = -1000000.0;
        for (skills, 0..) |skill, i| {
            if (taken[i]) continue;
            const vector_rank = vectorRank(skills, skill.vector_score);
            const sparse_component = if (skill.sparse_rank == std.math.maxInt(usize)) 0.0 else 1.0 / @as(f64, @floatFromInt(60 + skill.sparse_rank + 1));
            const vector_component = 1.0 / @as(f64, @floatFromInt(60 + vector_rank + 1));
            const score = sparse_component + vector_component + skill.vector_score * 0.01;
            if (score > best_score) {
                best_score = score;
                best_index = i;
            }
        }
        if (best_index) |bi| {
            taken[bi] = true;
            const s = skills[bi];
            try selected.append(SkillRecord{
                .id = try app.allocator.dupe(u8, s.id),
                .name = try app.allocator.dupe(u8, s.name),
                .description = try app.allocator.dupe(u8, s.description),
                .trigger_json = try app.allocator.dupe(u8, s.trigger_json),
                .procedure_json = try app.allocator.dupe(u8, s.procedure_json),
                .embedding_json = try app.allocator.dupe(u8, s.embedding_json),
                .vector_score = s.vector_score,
                .sparse_rank = s.sparse_rank,
                .rrf_score = best_score,
            });
        }
    }
    freeSkillRecords(app.allocator, skills);
    return selected.toOwnedSlice();
}

fn vectorRank(skills: []SkillRecord, score: f64) usize {
    var rank: usize = 0;
    for (skills) |skill| if (skill.vector_score > score) rank += 1;
    return rank;
}

fn findRank(ids: [][]u8, id: []const u8) usize {
    for (ids, 0..) |candidate, i| if (std.mem.eql(u8, candidate, id)) return i;
    return std.math.maxInt(usize);
}

fn freeSkillRecords(allocator: Allocator, records: []SkillRecord) void {
    for (records) |*record| record.deinit(allocator);
    allocator.free(records);
}

fn selectedSkillsJson(allocator: Allocator, skills: []SkillRecord) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    const w = out.writer();
    try w.writeAll("[");
    for (skills, 0..) |skill, i| {
        if (i != 0) try w.writeAll(",");
        try w.writeAll("{\"id\":");
        try writeJsonString(w, skill.id);
        try w.writeAll(",\"name\":");
        try writeJsonString(w, skill.name);
        try w.writeAll(",\"description\":");
        try writeJsonString(w, skill.description);
        try w.writeAll(",\"trigger\":");
        try w.writeAll(skill.trigger_json);
        try w.writeAll(",\"procedure\":");
        try w.writeAll(skill.procedure_json);
        try w.print(",\"retrieval_score\":{}}}", .{skill.rrf_score});
    }
    try w.writeAll("]");
    return out.toOwnedSlice();
}

fn buildStepPrompt(allocator: Allocator, procedure_json: []const u8, state_json: []const u8, observation_json: []const u8, skills_json: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    const w = out.writer();
    try w.writeAll("{\"runtime\":\"autonomous_state_agent\",\"contract\":{\"prompt_footprint\":\"O(1)\",\"history_policy\":\"No append-only conversation history or reasoning logs permitted.\",\"state_patch_semantics\":\"Top-level merge. Null values delete keys.\",\"output\":\"Return exactly one JSON object with keys state_patch, action, terminal, confidence.\"},\"procedure\":");
    try w.writeAll(procedure_json);
    try w.writeAll(",\"working_memory_state\":");
    try w.writeAll(state_json);
    try w.writeAll(",\"latest_environment_observation\":");
    try w.writeAll(observation_json);
    try w.writeAll(",\"experiential_memory_skills\":");
    try w.writeAll(skills_json);
    try w.writeAll(",\"allowed_actions\":[\"none\",\"finish\",\"sleep\",\"emit\",\"filesystem.write_file\",\"filesystem.read_file\",\"filesystem.append_file\",\"filesystem.replace_lines\",\"filesystem.check_lines\",\"filesystem.list_dir\",\"filesystem.delete_file\",\"compute\",\"memory.search\",\"memory.add_skill\",\"http.get\"],\"required_schema\":{\"state_patch\":\"object\",\"action\":{\"type\":\"string\",\"args\":\"object\"},\"terminal\":\"boolean\",\"confidence\":\"number\"}}");
    return out.toOwnedSlice();
}

fn buildCognitionFrame(allocator: Allocator, state_json: []const u8, observation_json: []const u8, step: i64) ![]u8 {
    var combined = std.ArrayList(u8).init(allocator);
    defer combined.deinit();
    try combined.writer().print("{s}\n{s}", .{ state_json, observation_json });
    var vector: [EMBED_DIM]f64 = undefined;
    embedText(combined.items, &vector);
    var out = std.ArrayList(u8).init(allocator);
    const w = out.writer();
    try w.print("{{\"step\":{},\"generated_at\":{},\"matrix\":[", .{ step, nowMillis() });
    var k: usize = 0;
    while (k < COG_K) : (k += 1) {
        if (k != 0) try w.writeAll(",");
        try w.writeAll("[");
        var h: usize = 0;
        while (h < COG_H) : (h += 1) {
            if (h != 0) try w.writeAll(",");
            try w.print("{d}", .{vector[k * COG_H + h]});
        }
        try w.writeAll("]");
    }
    const gate = if (containsIgnoreCase(state_json, "\"done\":true")) "verify" else "continue";
    try w.writeAll("],\"transition_gate\":");
    try writeJsonString(w, gate);
    try w.writeAll("}");
    return out.toOwnedSlice();
}

fn callModelStreaming(app: *App, user_prompt: []const u8) !ModelResult {
    const system_prompt = "You are an autonomous runtime policy. Reason privately, discard intermediate chain of thought, and output valid plain JSON matching the schema.";
    var client = std.http.Client{ .allocator = app.allocator };
    defer client.deinit();

    const url = try joinUrl(app.allocator, app.config.modular_base_url, "chat/completions");
    defer app.allocator.free(url);
    const uri = try std.Uri.parse(url);

    var body_buf = std.ArrayList(u8).init(app.allocator);
    defer body_buf.deinit();
    const bw = body_buf.writer();
    try bw.writeAll("{\"model\":");
    try writeJsonString(bw, app.config.model);
    try bw.writeAll(",\"messages\":[{\"role\":\"system\",\"content\":");
    try writeJsonString(bw, system_prompt);
    try bw.writeAll("},{\"role\":\"user\",\"content\":");
    try writeJsonString(bw, user_prompt);
    try bw.print("}],\"stream\":true,\"stream_options\":{{\"include_usage\":true}},\"temperature\":0.7,\"max_tokens\":{},\"response_format\":{{\"type\":\"json_object\"}}}}", .{app.config.model_max_tokens});

    const auth = try std.fmt.allocPrint(app.allocator, "Bearer {s}", .{app.config.modular_api_key});
    defer app.allocator.free(auth);

    var header_buf: [8192]u8 = undefined;
    var req = try client.open(.POST, uri, .{
        .server_header_buffer = &header_buf,
        .extra_headers = &.{
            .{ .name = "authorization", .value = auth },
            .{ .name = "content-type", .value = "application/json" },
            .{ .name = "accept", .value = "text/event-stream" },
        },
    });
    defer req.deinit();

    req.transfer_encoding = .{ .content_length = body_buf.items.len };
    try req.send();
    try req.writeAll(body_buf.items);
    try req.finish();
    try req.wait();

    if (req.response.status != .ok) return error.ModelRequestFailed;
    return parseModelResponseStream(app.allocator, req.reader());
}

fn parseModelResponseStream(allocator: Allocator, reader: anytype) !ModelResult {
    var content = std.ArrayList(u8).init(allocator);
    var total_tokens: i64 = 0;
    var line_buf = std.ArrayList(u8).init(allocator);
    defer line_buf.deinit();
    var chunk: [4096]u8 = undefined;

    while (true) {
        const n = reader.read(&chunk) catch break;
        if (n == 0) break;
        try line_buf.appendSlice(chunk[0..n]);
        while (std.mem.indexOfScalar(u8, line_buf.items, '\n')) |nl| {
            const raw_line = line_buf.items[0..nl];
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (std.mem.startsWith(u8, line, "data:")) {
                const payload = std.mem.trim(u8, line[5..], " \t");
                if (!std.mem.eql(u8, payload, "[DONE]")) {
                    var parsed = std.json.parseFromSlice(std.json.Value, allocator, payload, .{}) catch {
                        const rem = line_buf.items.len - (nl + 1);
                        std.mem.copyForwards(u8, line_buf.items[0..rem], line_buf.items[nl + 1 ..]);
                        line_buf.shrinkRetainingCapacity(rem);
                        continue;
                    };
                    defer parsed.deinit();
                    if (objectGetValue(parsed.value, "usage")) |usage| {
                        if (objectGetInt(usage, "total_tokens")) |t| total_tokens = t;
                    }
                    if (objectGetValue(parsed.value, "choices")) |choices| {
                        switch (choices) {
                            .array => |arr| {
                                if (arr.items.len > 0) {
                                    if (objectGetValue(arr.items[0], "delta")) |delta| {
                                        if (objectGetString(delta, "content")) |s| try content.appendSlice(s);
                                    }
                                }
                            },
                            else => {},
                        }
                    }
                }
            }
            const rem = line_buf.items.len - (nl + 1);
            std.mem.copyForwards(u8, line_buf.items[0..rem], line_buf.items[nl + 1 ..]);
            line_buf.shrinkRetainingCapacity(rem);
        }
    }
    return ModelResult{ .content = try content.toOwnedSlice(), .total_tokens = total_tokens };
}

fn validateEnvelope(app: *App, model_content: []const u8) !Envelope {
    const extracted = try extractJsonObject(app.allocator, model_content);
    defer app.allocator.free(extracted);
    var parsed = try std.json.parseFromSlice(std.json.Value, app.allocator, extracted, .{});
    defer parsed.deinit();
    try validateValueBounded(parsed.value, 0);
    const patch_value = objectGetValue(parsed.value, "state_patch") orelse return error.MissingStatePatch;
    const action_value = objectGetValue(parsed.value, "action") orelse return error.MissingAction;
    try validatePatchValue(patch_value);
    try validateActionValue(action_value);
    const patch_json = try jsonValueToOwned(app.allocator, patch_value);
    const action_json = try jsonValueToOwned(app.allocator, action_value);
    const envelope_json = try jsonValueToOwned(app.allocator, parsed.value);
    return Envelope{
        .envelope_json = envelope_json,
        .patch_json = patch_json,
        .action_json = action_json,
        .terminal = objectGetBool(parsed.value, "terminal") orelse false,
        .confidence = objectGetFloat(parsed.value, "confidence") orelse 0.0,
    };
}

fn extractJsonObject(allocator: Allocator, text: []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len >= 2 and trimmed[0] == '{' and trimmed[trimmed.len - 1] == '}') return allocator.dupe(u8, trimmed);
    var start: ?usize = null;
    var depth: i64 = 0;
    var in_string = false;
    var escaped = false;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        const ch = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            if (depth == 0) start = i;
            depth += 1;
        } else if (ch == '}') {
            depth -= 1;
            if (depth == 0 and start != null) return allocator.dupe(u8, text[start.? .. i + 1]);
        }
    }
    return error.NoJsonObject;
}

fn validateActionValue(value: std.json.Value) !void {
    const typ = objectGetString(value, "type") orelse return error.ActionMissingType;
    if (std.mem.eql(u8, typ, "none") or std.mem.eql(u8, typ, "finish") or std.mem.eql(u8, typ, "sleep") or std.mem.eql(u8, typ, "emit") or std.mem.eql(u8, typ, "filesystem.write_file") or std.mem.eql(u8, typ, "filesystem.read_file") or std.mem.eql(u8, typ, "filesystem.append_file") or std.mem.eql(u8, typ, "filesystem.replace_lines") or std.mem.eql(u8, typ, "filesystem.check_lines") or std.mem.eql(u8, typ, "filesystem.list_dir") or std.mem.eql(u8, typ, "filesystem.delete_file") or std.mem.eql(u8, typ, "compute") or std.mem.eql(u8, typ, "memory.search") or std.mem.eql(u8, typ, "memory.add_skill") or std.mem.eql(u8, typ, "http.get")) {
        try validateValueBounded(value, 0);
        return;
    }
    return error.ActionNotAllowed;
}

fn executeAuthorizedAction(app: *App, run_id: []const u8, step: i64, action_json: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, app.allocator, action_json, .{});
    defer parsed.deinit();
    const typ = objectGetString(parsed.value, "type") orelse return error.ActionMissingType;
    const args = objectGetValue(parsed.value, "args") orelse parsed.value;
    const generated_at = try app.db.latestCognitionGeneratedAt(run_id);
    const elapsed_ms = nowMillis() - generated_at;

    var staleness_buf = std.ArrayList(u8).init(app.allocator);
    defer staleness_buf.deinit();
    const sw = staleness_buf.writer();
    try sw.writeAll("[");
    var k: usize = 0;
    while (k < 4) : (k += 1) {
        const freq = std.math.pow(f64, 10000.0, @as(f64, @floatFromInt(k)) / 4.0);
        const t = @as(f64, @floatFromInt(elapsed_ms));
        if (k != 0) try sw.writeAll(",");
        try sw.print("{{\"sin\":{d:.4},\"cos\":{d:.4}}}", .{ @sin(t / freq), @cos(t / freq) });
    }
    try sw.writeAll("]");

    if (std.mem.eql(u8, typ, "none")) {
        return std.fmt.allocPrint(app.allocator, "{{\"ok\":true,\"type\":\"none\",\"step\":{},\"staleness\":{s},\"at\":{}}}", .{ step, staleness_buf.items, now() });
    }
    if (std.mem.eql(u8, typ, "finish")) {
        const result_value = objectGetValue(args, "result") orelse args;
        const result_json = try jsonValueToOwned(app.allocator, result_value);
        defer app.allocator.free(result_json);
        try app.db.markCompleted(run_id, result_json);
        return std.fmt.allocPrint(app.allocator, "{{\"ok\":true,\"type\":\"finish\",\"result\":{s},\"step\":{},\"staleness\":{s},\"at\":{}}}", .{ result_json, step, staleness_buf.items, now() });
    }
    if (std.mem.eql(u8, typ, "sleep")) {
        const ms = objectGetInt(args, "ms") orelse 1000;
        const clamped_ms = @min(@max(ms, 0), 60000);
        std.time.sleep(@as(u64, @intCast(clamped_ms)) * std.time.ns_per_ms);
        return std.fmt.allocPrint(app.allocator, "{{\"ok\":true,\"type\":\"sleep\",\"ms\":{},\"step\":{},\"staleness\":{s},\"at\":{}}}", .{ clamped_ms, step, staleness_buf.items, now() });
    }
    if (std.mem.eql(u8, typ, "emit")) {
        const payload_value = objectGetValue(args, "payload") orelse args;
        const payload_json = try jsonValueToOwned(app.allocator, payload_value);
        defer app.allocator.free(payload_json);
        var run = try app.db.getRun(run_id);
        defer run.deinit(app.allocator);
        _ = try app.db.insertEvent(run.tenant_id, run_id, step, "agent_emit", payload_json);
        return std.fmt.allocPrint(app.allocator, "{{\"ok\":true,\"type\":\"emit\",\"payload\":{s},\"step\":{},\"staleness\":{s},\"at\":{}}}", .{ payload_json, step, staleness_buf.items, now() });
    }
    if (std.mem.eql(u8, typ, "compute")) {
        const expr = objectGetString(args, "expression") orelse return error.MissingExpression;
        var ev = ExprEval{ .src = expr, .pos = 0 };
        const val = try ev.parseExpr();
        ev.skipWs();
        if (ev.pos != expr.len) return error.TrailingTokens;
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"compute\",\"expression\":");
        try writeJsonString(w, expr);
        try w.print(",\"result\":{d},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ val, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "memory.search")) {
        const query = objectGetString(args, "query") orelse return error.MissingQuery;
        var run = try app.db.getRun(run_id);
        defer run.deinit(app.allocator);
        const skills = try routeSkills(app, run.tenant_id, query, "{}");
        defer freeSkillRecords(app.allocator, skills);
        const sj = try selectedSkillsJson(app.allocator, skills);
        defer app.allocator.free(sj);
        return std.fmt.allocPrint(app.allocator, "{{\"ok\":true,\"type\":\"memory.search\",\"skills\":{s},\"step\":{},\"staleness\":{s},\"at\":{}}}", .{ sj, step, staleness_buf.items, now() });
    }
    if (std.mem.eql(u8, typ, "memory.add_skill")) {
        var run = try app.db.getRun(run_id);
        defer run.deinit(app.allocator);
        const skill_id = try makeId(app.allocator, "skill");
        defer app.allocator.free(skill_id);
        const name = objectGetString(args, "name") orelse return error.SkillMissingName;
        const description = objectGetString(args, "description") orelse return error.SkillMissingDescription;
        const trigger_value = objectGetValue(args, "trigger") orelse std.json.Value{ .object = std.json.ObjectMap.init(app.allocator) };
        const procedure_value = objectGetValue(args, "procedure") orelse return error.SkillMissingProcedure;
        const trigger_json = try jsonValueToOwned(app.allocator, trigger_value);
        defer app.allocator.free(trigger_json);
        const procedure_json = try jsonValueToOwned(app.allocator, procedure_value);
        defer app.allocator.free(procedure_json);
        try app.db.addSkillFromFields(run.tenant_id, skill_id, name, description, trigger_json, procedure_json, true);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"memory.add_skill\",\"skill_id\":");
        try writeJsonString(w, skill_id);
        try w.print(",\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.write_file")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        const content = objectGetString(args, "content") orelse return error.MissingContent;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        try atomicWriteFile(app.allocator, resolved, content);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.write_file\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"bytes_written\":{d},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ content.len, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.read_file")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        const data = try std.fs.cwd().readFileAlloc(app.allocator, resolved, 16 * 1024 * 1024);
        defer app.allocator.free(data);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.read_file\",\"path\":");
        try writeJsonString(w, path);
        try w.writeAll(",\"content\":");
        try writeJsonString(w, data);
        try w.print(",\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.append_file")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        const content = objectGetString(args, "content") orelse return error.MissingContent;
        const unique = objectGetBool(args, "unique") orelse false;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        const appended = try appendFileLineOriented(app.allocator, resolved, content, unique);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.append_file\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"lines_appended\":{d},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ appended, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.replace_lines")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        const start_line = objectGetInt(args, "start_line") orelse return error.MissingLineRange;
        const end_line = objectGetInt(args, "end_line") orelse return error.MissingLineRange;
        const content = objectGetString(args, "content") orelse return error.MissingContent;
        if (start_line < 1 or end_line < start_line) return error.InvalidLineRange;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        try replaceLines(app.allocator, resolved, @intCast(start_line), @intCast(end_line), content);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.replace_lines\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"start_line\":{d},\"end_line\":{d},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ start_line, end_line, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.check_lines")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        const lines_value = objectGetValue(args, "lines") orelse return error.MissingLines;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        const result = try checkLines(app.allocator, resolved, lines_value);
        defer app.allocator.free(result);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.check_lines\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"result\":{s},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ result, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.list_dir")) {
        var path = objectGetString(args, "path") orelse ".";
        if (path.len == 0) path = ".";
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        var dir = try std.fs.cwd().openDir(resolved, .{ .iterate = true });
        defer dir.close();
        var list = std.ArrayList(u8).init(app.allocator);
        defer list.deinit();
        const lw = list.writer();
        try lw.writeAll("[");
        var first = true;
        var it = dir.iterate();
        while (try it.next()) |entry| {
            if (!first) try lw.writeAll(",");
            first = false;
            try lw.writeAll("{\"name\":");
            try writeJsonString(lw, entry.name);
            try lw.writeAll(",\"is_dir\":");
            try lw.writeAll(if (entry.kind == .directory) "true" else "false");
            try lw.writeAll("}");
        }
        try lw.writeAll("]");
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.list_dir\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"entries\":{s},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ list.items, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.delete_file")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        try std.fs.cwd().deleteFile(resolved);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.delete_file\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "http.get")) {
        const url = objectGetString(args, "url") orelse return error.MissingUrl;
        if (!isHttpHostAllowed(app.config.allowed_http_hosts, url)) return error.HttpHostNotAllowed;
        var client = std.http.Client{ .allocator = app.allocator };
        defer client.deinit();
        const uri = try std.Uri.parse(url);
        var header_buf: [4096]u8 = undefined;
        var req = try client.open(.GET, uri, .{ .server_header_buffer = &header_buf });
        defer req.deinit();
        try req.send();
        try req.finish();
        try req.wait();
        const body = try req.reader().readAllAlloc(app.allocator, 1024 * 1024);
        defer app.allocator.free(body);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"http.get\",\"url\":");
        try writeJsonString(w, url);
        try w.writeAll(",\"body\":");
        try writeJsonString(w, body);
        try w.print(",\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    return error.ActionNotAllowed;
}

fn evaluateStepVerifier(allocator: Allocator, state_json: []const u8, outcome_json: []const u8) ![]u8 {
    const success = !containsIgnoreCase(outcome_json, "\"ok\":false") and !containsIgnoreCase(state_json, "\"done\":false");
    return std.fmt.allocPrint(allocator, "{{\"success\":{},\"checked_at\":{},\"scope\":\"step\"}}", .{ success, now() });
}

fn finalizeTrajectory(app: *App, run_id: []const u8) !void {
    if (!try app.db.claimFinalization(run_id)) return;
    var run = app.db.getRun(run_id) catch return;
    defer run.deinit(app.allocator);
    const verifier = try evaluateTerminalVerifiers(app, &run);
    defer app.allocator.free(verifier);
    try app.db.insertEvent(run.tenant_id, run_id, run.step, "terminal_verifier", verifier);
    if (containsIgnoreCase(verifier, "\"success\":false")) {
        const reflection = generateReflectionPatch(app, &run, verifier) catch |err| blk: {
            break :blk try std.fmt.allocPrint(app.allocator, "{{\"diagnosis\":\"reflection_generation_failed\",\"error\":\"{s}\",\"patches\":[]}}", .{@errorName(err)});
        };
        defer app.allocator.free(reflection);
        try app.db.insertEvent(run.tenant_id, run_id, run.step, "reflection_patch", reflection);
        const patch_id = try makeId(app.allocator, "patch");
        defer app.allocator.free(patch_id);
        const report = validateAndApplySkillPatch(app, run.tenant_id, reflection) catch |err| blk: {
            break :blk try std.fmt.allocPrint(app.allocator, "{{\"accepted\":false,\"error\":\"{s}\"}}", .{@errorName(err)});
        };
        defer app.allocator.free(report);
        try app.db.insertSkillPatch(run.tenant_id, patch_id, null, if (containsIgnoreCase(report, "\"accepted\":true")) "accepted" else "rejected", reflection, report);
        optimizePolicyFromReflection(app, &run, reflection) catch {};
    }
}

fn evaluateTerminalVerifiers(app: *App, run: *RunRecord) ![]u8 {
    const state = try app.db.getStateJson(run.id);
    defer app.allocator.free(state);
    var parsed = std.json.parseFromSlice(std.json.Value, app.allocator, run.procedure_json, .{}) catch {
        const success = containsIgnoreCase(state, "\"done\":true") or std.mem.eql(u8, run.status, "completed");
        return std.fmt.allocPrint(app.allocator, "{{\"success\":{},\"scope\":\"terminal\",\"checked_at\":{}}}", .{ success, now() });
    };
    defer parsed.deinit();

    var all_passed = true;
    var details = std.ArrayList(u8).init(app.allocator);
    defer details.deinit();
    const dw = details.writer();
    try dw.writeAll("[");
    var has_verifiers = false;

    if (objectGetValue(parsed.value, "verifiers")) |verifiers| {
        switch (verifiers) {
            .array => |arr| {
                for (arr.items, 0..) |v, idx| {
                    has_verifiers = true;
                    if (idx != 0) try dw.writeAll(",");
                    const vtype = objectGetString(v, "type") orelse "unknown";
                    var passed = false;
                    if (std.mem.eql(u8, vtype, "state_key_exists")) {
                        if (objectGetString(v, "key")) |k| passed = stateHasKey(app.allocator, state, k);
                    } else if (std.mem.eql(u8, vtype, "state_key_equals")) {
                        if (objectGetString(v, "key")) |k| {
                            if (objectGetValue(v, "value")) |expected| passed = stateKeyEquals(app.allocator, state, k, expected);
                        }
                    } else if (std.mem.eql(u8, vtype, "file_exists")) {
                        if (objectGetString(v, "path")) |p| {
                            const resolved = resolveWorkspacePath(app.allocator, app.config.workspace_root, p) catch "";
                            if (resolved.len > 0) {
                                defer app.allocator.free(resolved);
                                passed = blk: {
                                    std.fs.cwd().access(resolved, .{}) catch break :blk false;
                                    break :blk true;
                                };
                            }
                        }
                    } else if (std.mem.eql(u8, vtype, "file_contains_lines")) {
                        if (objectGetString(v, "path")) |p| {
                            const lines = objectGetValue(v, "lines") orelse std.json.Value{ .array = std.json.Array.init(app.allocator) };
                            const resolved = resolveWorkspacePath(app.allocator, app.config.workspace_root, p) catch "";
                            if (resolved.len > 0) {
                                defer app.allocator.free(resolved);
                                const check_res = checkLines(app.allocator, resolved, lines) catch "[]";
                                if (check_res.ptr != "[]".ptr) defer app.allocator.free(check_res);
                                passed = !containsIgnoreCase(check_res, "false");
                            }
                        }
                    } else if (std.mem.eql(u8, vtype, "min_progress")) {
                        const min_val = objectGetFloat(v, "value") orelse 1.0;
                        var state_parsed = std.json.parseFromSlice(std.json.Value, app.allocator, state, .{}) catch null;
                        if (state_parsed) |*sp| {
                            defer sp.deinit();
                            const actual_prog = objectGetFloat(sp.value, "progress") orelse 0.0;
                            passed = actual_prog >= min_val;
                        }
                    }
                    if (!passed) all_passed = false;
                    try dw.writeAll("{\"type\":");
                    try writeJsonString(dw, vtype);
                    try dw.print(",\"passed\":{}}}", .{passed});
                }
            },
            else => {},
        }
    }
    try dw.writeAll("]");

    if (!has_verifiers) {
        all_passed = containsIgnoreCase(state, "\"done\":true") or std.mem.eql(u8, run.status, "completed");
    }

    return std.fmt.allocPrint(app.allocator, "{{\"success\":{},\"scope\":\"terminal\",\"details\":{s},\"checked_at\":{}}}", .{ all_passed, details.items, now() });
}

fn stateHasKey(allocator: Allocator, state_json: []const u8, key: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, state_json, .{}) catch return false;
    defer parsed.deinit();
    return objectGetValue(parsed.value, key) != null;
}

fn stateKeyEquals(allocator: Allocator, state_json: []const u8, key: []const u8, expected: std.json.Value) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, state_json, .{}) catch return false;
    defer parsed.deinit();
    const actual = objectGetValue(parsed.value, key) orelse return false;
    const a = jsonValueToOwned(allocator, actual) catch return false;
    defer allocator.free(a);
    const e = jsonValueToOwned(allocator, expected) catch return false;
    defer allocator.free(e);
    return std.mem.eql(u8, a, e);
}

fn generateReflectionPatch(app: *App, run: *RunRecord, verifier_json: []const u8) ![]u8 {
    const state = try app.db.getStateJson(run.id);
    defer app.allocator.free(state);
    var prompt = std.ArrayList(u8).init(app.allocator);
    defer prompt.deinit();
    try prompt.writer().print("Generate compact JSON reflection patch. Output JSON only with keys diagnosis, failure_component, pivot_actions, skill_patch. Procedure: {s}\nTerminal state: {s}\nVerifier: {s}", .{ run.procedure_json, state, verifier_json });
    var result = try callModelStreaming(app, prompt.items);
    defer result.deinit(app.allocator);
    return extractJsonObject(app.allocator, result.content);
}

fn validateAndApplySkillPatch(app: *App, tenant_id: []const u8, patch_json: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, app.allocator, patch_json, .{});
    defer parsed.deinit();
    try validateValueBounded(parsed.value, 0);
    const skill_patch = objectGetValue(parsed.value, "skill_patch") orelse parsed.value;
    const skill = objectGetValue(skill_patch, "skill") orelse return error.MissingSkill;
    const skill_id = if (objectGetString(skill, "id")) |sid| try app.allocator.dupe(u8, sid) else try makeId(app.allocator, "skill");
    defer app.allocator.free(skill_id);
    const name = objectGetString(skill, "name") orelse return error.SkillMissingName;
    const description = objectGetString(skill, "description") orelse return error.SkillMissingDescription;
    const trigger_value = objectGetValue(skill, "trigger") orelse std.json.Value{ .object = std.json.ObjectMap.init(app.allocator) };
    const procedure_value = objectGetValue(skill, "procedure") orelse return error.SkillMissingProcedure;
    const trigger_json = try jsonValueToOwned(app.allocator, trigger_value);
    defer app.allocator.free(trigger_json);
    const procedure_json = try jsonValueToOwned(app.allocator, procedure_value);
    defer app.allocator.free(procedure_json);

    try app.db.addSkillFromFields(tenant_id, skill_id, name, description, trigger_json, procedure_json, true);
    var out = std.ArrayList(u8).init(app.allocator);
    errdefer out.deinit();
    const w = out.writer();
    try w.writeAll("{\"accepted\":true,\"skill_id\":");
    try writeJsonString(w, skill_id);
    try w.print(",\"validated_at\":{d}}}", .{now()});
    return out.toOwnedSlice();
}

fn optimizePolicyFromReflection(app: *App, run: *RunRecord, reflection_json: []const u8) !void {
    const state = try app.db.getStateJson(run.id);
    defer app.allocator.free(state);
    const clean_prompt = try buildStepPrompt(app.allocator, run.procedure_json, state, run.latest_observation_json, "[]");
    defer app.allocator.free(clean_prompt);
    const privileged_prompt = try std.fmt.allocPrint(app.allocator, "{{\"reflection_patch\":{s},\"clean_prompt\":{s}}}", .{ reflection_json, clean_prompt });
    defer app.allocator.free(privileged_prompt);
    const teacher = try callModelLogprobs(app, privileged_prompt);
    defer app.allocator.free(teacher);
    const student = try callModelLogprobs(app, clean_prompt);
    defer app.allocator.free(student);
    const kl = computeReverseKl(app.allocator, teacher, student) catch 0.0;
    try app.db.insertDistillationExample(run.tenant_id, run.id, reflection_json, clean_prompt, teacher, student, kl);
}

fn callModelLogprobs(app: *App, user_prompt: []const u8) ![]u8 {
    var client = std.http.Client{ .allocator = app.allocator };
    defer client.deinit();
    const url = try joinUrl(app.allocator, app.config.modular_base_url, "chat/completions");
    defer app.allocator.free(url);
    const uri = try std.Uri.parse(url);

    var body_buf = std.ArrayList(u8).init(app.allocator);
    defer body_buf.deinit();
    const bw = body_buf.writer();
    try bw.writeAll("{\"model\":");
    try writeJsonString(bw, app.config.model);
    try bw.writeAll(",\"messages\":[{\"role\":\"user\",\"content\":");
    try writeJsonString(bw, user_prompt);
    try bw.writeAll("}],\"stream\":false,\"temperature\":0.7,\"max_tokens\":2048,\"logprobs\":true,\"top_logprobs\":5,\"response_format\":{\"type\":\"json_object\"}}");

    const auth = try std.fmt.allocPrint(app.allocator, "Bearer {s}", .{app.config.modular_api_key});
    defer app.allocator.free(auth);

    var header_buf: [8192]u8 = undefined;
    var req = try client.open(.POST, uri, .{
        .server_header_buffer = &header_buf,
        .extra_headers = &.{
            .{ .name = "authorization", .value = auth },
            .{ .name = "content-type", .value = "application/json" },
        },
    });
    defer req.deinit();

    req.transfer_encoding = .{ .content_length = body_buf.items.len };
    try req.send();
    try req.writeAll(body_buf.items);
    try req.finish();
    try req.wait();

    if (req.response.status != .ok) return error.ModelRequestFailed;
    const body = try req.reader().readAllAlloc(app.allocator, app.config.max_response_bytes);
    defer app.allocator.free(body);

    var parsed = try std.json.parseFromSlice(std.json.Value, app.allocator, body, .{});
    defer parsed.deinit();
    if (objectGetValue(parsed.value, "choices")) |choices| switch (choices) {
        .array => |arr| {
            if (arr.items.len > 0) {
                if (objectGetValue(arr.items[0], "logprobs")) |lp| return jsonValueToOwned(app.allocator, lp);
            }
        },
        else => {},
    };
    return app.allocator.dupe(u8, "{}");
}

fn computeReverseKl(allocator: Allocator, teacher_json: []const u8, student_json: []const u8) !f64 {
    var teacher = try std.json.parseFromSlice(std.json.Value, allocator, teacher_json, .{});
    defer teacher.deinit();
    var student = try std.json.parseFromSlice(std.json.Value, allocator, student_json, .{});
    defer student.deinit();
    const tc = objectGetValue(teacher.value, "content") orelse return 0.0;
    const sc = objectGetValue(student.value, "content") orelse return 0.0;
    var kl: f64 = 0.0;
    switch (tc) {
        .array => |ta| switch (sc) {
            .array => |sa| {
                const n = @min(ta.items.len, sa.items.len);
                var i: usize = 0;
                while (i < n) : (i += 1) {
                    const tops = objectGetValue(ta.items[i], "top_logprobs") orelse continue;
                    switch (tops) {
                        .array => |toparr| {
                            for (toparr.items) |top| {
                                const tok = objectGetString(top, "token") orelse continue;
                                const logq = objectGetFloat(top, "logprob") orelse continue;
                                const logp = findLogprob(sa.items[i], tok) orelse -30.0;
                                const q = std.math.exp(logq);
                                kl += q * (logq - logp);
                            }
                        },
                        else => {},
                    }
                }
            },
            else => {},
        },
        else => {},
    }
    return kl;
}

fn findLogprob(item: std.json.Value, token: []const u8) ?f64 {
    const tops = objectGetValue(item, "top_logprobs") orelse return null;
    switch (tops) {
        .array => |arr| {
            for (arr.items) |top| {
                const tok = objectGetString(top, "token") orelse continue;
                if (std.mem.eql(u8, tok, token)) return objectGetFloat(top, "logprob");
            }
        },
        else => {},
    }
    return null;
}

fn metaHarnessEntry(app: *App) void {
    while (true) {
        consolidateKnowledge(app, app.config.default_tenant_id) catch {};
        std.time.sleep(300 * std.time.ns_per_s);
    }
}

fn consolidateKnowledge(app: *App, tenant_id: []const u8) !void {
    const markdown = try app.db.buildKnowledgeMarkdown(tenant_id);
    defer app.allocator.free(markdown);
    try std.fs.cwd().makePath(app.config.knowledge_root);
    const playbook_path = try std.fs.path.join(app.allocator, &.{ app.config.knowledge_root, "PLAYBOOK.md" });
    defer app.allocator.free(playbook_path);
    const old = std.fs.cwd().readFileAlloc(app.allocator, playbook_path, 16 * 1024 * 1024) catch "";
    const had_old = old.len > 0 and old.ptr != "".ptr;
    defer if (had_old) app.allocator.free(old);
    try atomicWriteFile(app.allocator, playbook_path, markdown);
    _ = runGit(app, &[_][]const u8{ "git", "-C", app.config.knowledge_root, "init" }) catch {};
    _ = runGit(app, &[_][]const u8{ "git", "-C", app.config.knowledge_root, "add", "PLAYBOOK.md" }) catch {};
    const message = try std.fmt.allocPrint(app.allocator, "knowledge consolidation {}", .{now()});
    defer app.allocator.free(message);
    _ = runGit(app, &[_][]const u8{ "git", "-C", app.config.knowledge_root, "-c", "user.name=agent-runtime", "-c", "user.email=agent-runtime@local", "commit", "-m", message }) catch {};
    const commit = runGit(app, &[_][]const u8{ "git", "-C", app.config.knowledge_root, "rev-parse", "HEAD" }) catch try app.allocator.dupe(u8, "head");
    defer app.allocator.free(commit);
    const diff = diffText(app.allocator, old, markdown) catch try app.allocator.dupe(u8, "no diff");
    defer app.allocator.free(diff);
    try app.db.insertKnowledgeVersion(tenant_id, markdown, std.mem.trim(u8, commit, " \t\r\n"), diff);
}

fn runGit(app: *App, argv: []const []const u8) ![]u8 {
    const result = try std.process.Child.run(.{ .allocator = app.allocator, .argv = argv, .max_output_bytes = 1024 * 1024 });
    defer app.allocator.free(result.stderr);
    switch (result.term) {
        .Exited => |code| {
            if (code != 0) {
                app.allocator.free(result.stdout);
                return error.GitFailed;
            }
        },
        else => {
            app.allocator.free(result.stdout);
            return error.GitFailed;
        },
    }
    return result.stdout;
}

fn diffText(allocator: Allocator, old: []const u8, new: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    if (std.mem.eql(u8, old, new)) {
        try out.writer().writeAll("no changes\n");
        return out.toOwnedSlice();
    }
    var old_lines = std.mem.splitScalar(u8, old, '\n');
    var new_lines = std.mem.splitScalar(u8, new, '\n');
    var line: usize = 1;
    while (true) : (line += 1) {
        const o = old_lines.next();
        const n = new_lines.next();
        if (o == null and n == null) break;
        if (o == null) {
            try out.writer().print("+{}:{s}\n", .{ line, n.? });
        } else if (n == null) {
            try out.writer().print("-{}:{s}\n", .{ line, o.? });
        } else if (!std.mem.eql(u8, o.?, n.?)) {
            try out.writer().print("-{}:{s}\n+{}:{s}\n", .{ line, o.?, line, n.? });
        }
    }
    return out.toOwnedSlice();
}

fn startHttpServer(app: *App) !void {
    const address = try std.net.Address.parseIp(app.config.host, app.config.port);
    const fd = try std.posix.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    defer std.posix.close(fd);
    var yes: c_int = 1;
    try std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, std.mem.asBytes(&yes));
    try std.posix.bind(fd, &address.any, address.getOsSockLen());
    try std.posix.listen(fd, 128);
    while (true) {
        const client = try std.posix.accept(fd, null, null, 0);
        const thread = try std.Thread.spawn(.{}, connectionEntry, .{ app, client });
        thread.detach();
    }
}

fn connectionEntry(app: *App, fd: std.posix.socket_t) void {
    defer std.posix.close(fd);
    var arena = std.heap.ArenaAllocator.init(app.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var req = readHttpRequest(allocator, fd, app.config.max_request_bytes) catch return;
    handleRequest(app, allocator, fd, &req) catch return;
}

fn readHttpRequest(allocator: Allocator, fd: std.posix.socket_t, max_bytes: usize) !HttpRequest {
    var buf = std.ArrayList(u8).init(allocator);
    var temp: [8192]u8 = undefined;
    var header_end: ?usize = null;
    var content_length: usize = 0;
    while (true) {
        const n = try std.posix.read(fd, &temp);
        if (n == 0) break;
        try buf.appendSlice(temp[0..n]);
        if (buf.items.len > max_bytes) return error.RequestTooLarge;
        if (header_end == null) {
            if (std.mem.indexOf(u8, buf.items, "\r\n\r\n")) |idx| {
                header_end = idx;
                content_length = parseContentLength(buf.items[0..idx]);
            }
        }
        if (header_end) |he| {
            if (buf.items.len >= he + 4 + content_length) break;
        }
    }
    const raw = buf.items;
    const he = header_end orelse return error.BadRequest;
    const headers_part = raw[0..he];
    const body_start = he + 4;
    if (raw.len < body_start + content_length) return error.BadRequest;
    const body = raw[body_start .. body_start + content_length];
    var lines = std.mem.splitSequence(u8, headers_part, "\r\n");
    const start_line = lines.next() orelse return error.BadRequest;
    var parts = std.mem.tokenizeScalar(u8, start_line, ' ');
    const method = parts.next() orelse return error.BadRequest;
    const target = parts.next() orelse return error.BadRequest;
    var path = target;
    var query: []const u8 = "";
    if (std.mem.indexOfScalar(u8, target, '?')) |qidx| {
        path = target[0..qidx];
        query = target[qidx + 1 ..];
    }
    var headers = std.StringHashMap([]const u8).init(allocator);
    while (lines.next()) |line| {
        if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
            const key_raw = std.mem.trim(u8, line[0..colon], " \t");
            const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
            const key = try allocator.alloc(u8, key_raw.len);
            for (key_raw, 0..) |c, i| key[i] = std.ascii.toLower(c);
            try headers.put(key, value);
        }
    }
    return HttpRequest{ .method = method, .target = target, .path = path, .query = query, .headers = headers, .body = body };
}

fn parseContentLength(headers_part: []const u8) usize {
    var lines = std.mem.splitSequence(u8, headers_part, "\r\n");
    _ = lines.next();
    while (lines.next()) |line| {
        if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
            const key = std.mem.trim(u8, line[0..colon], " \t");
            if (std.ascii.eqlIgnoreCase(key, "content-length")) {
                const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
                return std.fmt.parseInt(usize, value, 10) catch 0;
            }
        }
    }
    return 0;
}

fn handleRequest(app: *App, allocator: Allocator, fd: std.posix.socket_t, req: *HttpRequest) !void {
    if (std.mem.eql(u8, req.method, "OPTIONS")) {
        try sendResponse(allocator, fd, 204, "text/plain", "");
        return;
    }
    const tenant_id = req.headers.get("x-tenant-id") orelse app.config.default_tenant_id;
    try app.db.ensureTenant(tenant_id);

    if (std.mem.eql(u8, req.method, "GET") and (std.mem.eql(u8, req.path, "/") or std.mem.eql(u8, req.path, "/index.html"))) {
        const index = std.fs.cwd().readFileAlloc(allocator, "index.html", 16 * 1024 * 1024) catch "<!doctype html><html><body><h1>Autonomous Agent Runtime</h1></body></html>";
        try sendResponse(allocator, fd, 200, "text/html; charset=utf-8", index);
        return;
    }
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/api/health")) {
        try sendResponse(allocator, fd, 200, "application/json", "{\"ok\":true,\"runtime\":\"autonomous-state-agent\"}");
        return;
    }
    if (std.mem.eql(u8, req.method, "POST") and (std.mem.eql(u8, req.path, "/api/runs") or std.mem.eql(u8, req.path, "/api/tasks") or std.mem.eql(u8, req.path, "/api/chat"))) {
        const response = try createRunOrChat(app, allocator, tenant_id, req.body);
        try sendResponse(allocator, fd, 200, "application/json", response);
        return;
    }
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/api/runs")) {
        const json = try app.db.listRunsJson(tenant_id);
        defer app.allocator.free(json);
        try sendResponse(allocator, fd, 200, "application/json", json);
        return;
    }
    if (std.mem.startsWith(u8, req.path, "/api/runs/")) {
        const rest = req.path["/api/runs/".len..];
        var segs = std.mem.splitScalar(u8, rest, '/');
        const run_id = segs.next() orelse return error.BadRequest;
        const tail = segs.next();
        if (tail == null and std.mem.eql(u8, req.method, "GET")) {
            const json = try app.db.runJson(tenant_id, run_id);
            defer app.allocator.free(json);
            try sendResponse(allocator, fd, 200, "application/json", json);
            return;
        }
        if (tail != null and std.mem.eql(u8, tail.?, "state") and std.mem.eql(u8, req.method, "GET")) {
            const state = try app.db.getStateJson(run_id);
            defer app.allocator.free(state);
            try sendResponse(allocator, fd, 200, "application/json", state);
            return;
        }
        if (tail != null and std.mem.eql(u8, tail.?, "events") and std.mem.eql(u8, req.method, "GET")) {
            try streamEvents(app, allocator, fd, tenant_id, run_id, req.query);
            return;
        }
        if (tail != null and std.mem.eql(u8, tail.?, "observe") and std.mem.eql(u8, req.method, "POST")) {
            try validateJsonText(allocator, req.body);
            try app.db.enqueueObservation(run_id, req.body);
            try app.db.insertEvent(tenant_id, run_id, 0, "external_observation", req.body);
            try sendResponse(allocator, fd, 200, "application/json", "{\"ok\":true}");
            return;
        }
        if (tail != null and std.mem.eql(u8, tail.?, "stop") and std.mem.eql(u8, req.method, "POST")) {
            try app.db.markRunStatus(run_id, "stopped", null);
            try sendResponse(allocator, fd, 200, "application/json", "{\"ok\":true}");
            return;
        }
        if (tail != null and std.mem.eql(u8, tail.?, "resume") and std.mem.eql(u8, req.method, "POST")) {
            try app.db.markRunStatus(run_id, "running", null);
            try spawnRun(app, run_id);
            try sendResponse(allocator, fd, 200, "application/json", "{\"ok\":true}");
            return;
        }
    }
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/api/skills/search")) {
        const q = try queryParam(allocator, req.query, "q");
        const skills = try routeSkills(app, tenant_id, q, "{}");
        defer freeSkillRecords(app.allocator, skills);
        const json = try selectedSkillsJson(allocator, skills);
        try sendResponse(allocator, fd, 200, "application/json", json);
        return;
    }
    if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/api/meta/consolidate")) {
        try consolidateKnowledge(app, tenant_id);
        try sendResponse(allocator, fd, 200, "application/json", "{\"ok\":true}");
        return;
    }
    try sendResponse(allocator, fd, 404, "application/json", "{\"error\":\"not_found\"}");
}

fn createRunOrChat(app: *App, allocator: Allocator, tenant_id: []const u8, body: []const u8) ![]u8 {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        const escaped_task = try jsonStringAlloc(allocator, body);
        const procedure = try std.fmt.allocPrint(allocator, "{{\"objective\":{s},\"max_steps\":{}}}", .{ escaped_task, app.config.max_steps });
        const initial = try std.fmt.allocPrint(allocator, "{{\"type\":\"user_task\",\"content\":{s}}}", .{escaped_task});
        const run_id = try makeId(app.allocator, "run");
        try app.db.createRun(tenant_id, run_id, procedure, initial, app.config.default_token_budget);
        try spawnRun(app, run_id);
        const run_json = try jsonStringAlloc(allocator, run_id);
        app.allocator.free(run_id);
        return std.fmt.allocPrint(allocator, "{{\"ok\":true,\"run_id\":{s},\"status\":\"running\"}}", .{run_json});
    };
    defer parsed.deinit();
    const task = objectGetString(parsed.value, "task") orelse objectGetString(parsed.value, "message") orelse "task";
    const task_json = try jsonStringAlloc(allocator, task);
    const procedure_json = try std.fmt.allocPrint(allocator, "{{\"objective\":{s},\"max_steps\":{}}}", .{ task_json, app.config.max_steps });
    const initial_observation = try std.fmt.allocPrint(allocator, "{{\"type\":\"user_task\",\"content\":{s}}}", .{task_json});
    const run_id = try makeId(app.allocator, "run");
    try app.db.createRun(tenant_id, run_id, procedure_json, initial_observation, app.config.default_token_budget);
    try spawnRun(app, run_id);
    const run_id_json = try jsonStringAlloc(allocator, run_id);
    app.allocator.free(run_id);
    return std.fmt.allocPrint(allocator, "{{\"ok\":true,\"run_id\":{s},\"status\":\"running\"}}", .{run_id_json});
}

fn streamEvents(app: *App, allocator: Allocator, fd: std.posix.socket_t, tenant_id: []const u8, run_id: []const u8, query: []const u8) !void {
    const header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    try writeAllFd(fd, header);
    var last_id_text = queryParam(allocator, query, "last_id") catch "";
    var last_id = std.fmt.parseInt(i64, last_id_text, 10) catch 0;
    var loops: usize = 0;
    while (loops < 3600) : (loops += 1) {
        const events = try app.db.eventsSinceJson(tenant_id, run_id, last_id);
        defer app.allocator.free(events);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, events, .{});
        defer parsed.deinit();
        switch (parsed.value) {
            .array => |arr| {
                for (arr.items) |event| {
                    const id = objectGetInt(event, "id") orelse last_id;
                    const typ = objectGetString(event, "type") orelse "message";
                    const payload = objectGetValue(event, "payload") orelse event;
                    const payload_json = try jsonValueToOwned(allocator, payload);
                    defer allocator.free(payload_json);
                    const chunk = try std.fmt.allocPrint(allocator, "id: {}\nevent: {s}\ndata: {s}\n\n", .{ id, typ, payload_json });
                    defer allocator.free(chunk);
                    try writeAllFd(fd, chunk);
                    last_id = id;
                }
            },
            else => {},
        }
        try writeAllFd(fd, ": keepalive\n\n");
        std.time.sleep(1000 * std.time.ns_per_ms);
    }
}

fn sendResponse(allocator: Allocator, fd: std.posix.socket_t, status: u16, content_type: []const u8, body: []const u8) !void {
    const reason = switch (status) {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        else => "OK",
    };
    const header = try std.fmt.allocPrint(allocator, "HTTP/1.1 {} {s}\r\nContent-Type: {s}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n", .{ status, reason, content_type, body.len });
    defer allocator.free(header);
    try writeAllFd(fd, header);
    try writeAllFd(fd, body);
}

fn writeAllFd(fd: std.posix.socket_t, data: []const u8) !void {
    var offset: usize = 0;
    while (offset < data.len) {
        const n = try std.posix.write(fd, data[offset..]);
        if (n == 0) return error.ConnectionClosed;
        offset += n;
    }
}

fn queryParam(allocator: Allocator, query: []const u8, name: []const u8) ![]u8 {
    var pairs = std.mem.splitScalar(u8, query, '&');
    while (pairs.next()) |pair| {
        if (std.mem.indexOfScalar(u8, pair, '=')) |eq| {
            if (std.mem.eql(u8, pair[0..eq], name)) return allocator.dupe(u8, pair[eq + 1 ..]);
        }
    }
    return allocator.dupe(u8, "");
}

fn writeJsonString(writer: anytype, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |ch| {
        switch (ch) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0...7, 11, 12, 14...31 => try writer.print("\\u{X:0>4}", .{ch}),
            else => try writer.writeByte(ch),
        }
    }
    try writer.writeByte('"');
}

fn jsonStringAlloc(allocator: Allocator, s: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    try writeJsonString(out.writer(), s);
    return out.toOwnedSlice();
}

fn jsonValueToOwned(allocator: Allocator, value: std.json.Value) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    try std.json.stringify(value, .{}, out.writer());
    return out.toOwnedSlice();
}

fn validateJsonText(allocator: Allocator, text: []const u8) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();
    try validateValueBounded(parsed.value, 0);
}

fn validateValueBounded(value: std.json.Value, depth: usize) !void {
    if (depth > 12) return error.JsonTooDeep;
    switch (value) {
        .object => |obj| {
            var count: usize = 0;
            var it = obj.iterator();
            while (it.next()) |entry| {
                count += 1;
                if (count > 256) return error.JsonObjectTooLarge;
                if (entry.key_ptr.*.len > 256) return error.JsonKeyTooLarge;
                if (containsBannedKey(entry.key_ptr.*)) return error.BannedStateKey;
                try validateValueBounded(entry.value_ptr.*, depth + 1);
            }
        },
        .array => |arr| {
            if (arr.items.len > 128) return error.JsonArrayTooLarge;
            for (arr.items) |item| try validateValueBounded(item, depth + 1);
        },
        .string => |s| {
            if (s.len > 32768) return error.JsonStringTooLarge;
            if (containsIgnoreCase(s, "chain_of_thought") or containsIgnoreCase(s, "reasoning trace")) return error.ReasoningLeak;
        },
        else => {},
    }
}

fn validatePatchValue(value: std.json.Value) !void {
    switch (value) {
        .object => {},
        else => return error.InvalidStatePatch,
    }
    try validateValueBounded(value, 0);
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (haystack.len < needle.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var j: usize = 0;
        while (j < needle.len) : (j += 1) {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(needle[j])) break;
        }
        if (j == needle.len) return true;
    }
    return false;
}

fn containsBannedKey(key: []const u8) bool {
    return std.ascii.eqlIgnoreCase(key, "history") or
        std.ascii.eqlIgnoreCase(key, "messages") or
        std.ascii.eqlIgnoreCase(key, "transcript") or
        std.ascii.eqlIgnoreCase(key, "chain_of_thought") or
        std.ascii.eqlIgnoreCase(key, "reasoning_trace") or
        std.ascii.eqlIgnoreCase(key, "scratchpad") or
        std.ascii.eqlIgnoreCase(key, "prior_actions") or
        std.ascii.eqlIgnoreCase(key, "past_reasoning") or
        std.ascii.eqlIgnoreCase(key, "raw_dialogue");
}

fn estimateTokens(bytes: usize) i64 {
    const v = (bytes + 3) / 4;
    return @intCast(v);
}

fn objectGetValue(value: std.json.Value, key: []const u8) ?std.json.Value {
    switch (value) {
        .object => |obj| return obj.get(key),
        else => return null,
    }
}

fn objectGetString(value: std.json.Value, key: []const u8) ?[]const u8 {
    const v = objectGetValue(value, key) orelse return null;
    switch (v) {
        .string => |s| return s,
        else => return null,
    }
}

fn objectGetBool(value: std.json.Value, key: []const u8) ?bool {
    const v = objectGetValue(value, key) orelse return null;
    switch (v) {
        .bool => |b| return b,
        else => return null,
    }
}

fn objectGetInt(value: std.json.Value, key: []const u8) ?i64 {
    const v = objectGetValue(value, key) orelse return null;
    switch (v) {
        .integer => |i| return i,
        .float => |f| return @intFromFloat(f),
        else => return null,
    }
}

fn objectGetFloat(value: std.json.Value, key: []const u8) ?f64 {
    const v = objectGetValue(value, key) orelse return null;
    switch (v) {
        .integer => |i| return @floatFromInt(i),
        .float => |f| return f,
        else => return null,
    }
}

fn embedText(text: []const u8, out: *[EMBED_DIM]f64) void {
    for (out) |*v| v.* = 0.0;
    var hash: u64 = 14695981039346656037;
    var active = false;
    var count: usize = 0;
    for (text) |c0| {
        const c = std.ascii.toLower(c0);
        if (std.ascii.isAlphanumeric(c)) {
            active = true;
            count += 1;
            hash ^= c;
            hash *%= 1099511628211;
        } else if (active) {
            commitEmbeddingToken(out, hash, count);
            hash = 14695981039346656037;
            active = false;
            count = 0;
        }
    }
    if (active) commitEmbeddingToken(out, hash, count);
    var norm: f64 = 0.0;
    for (out.*) |v| norm += v * v;
    norm = std.math.sqrt(norm);
    if (norm > 0.0) for (out) |*v| v.* /= norm;
}

fn commitEmbeddingToken(out: *[EMBED_DIM]f64, hash: u64, count: usize) void {
    const idx: usize = @intCast(hash % EMBED_DIM);
    const sign: f64 = if ((hash & 1) == 0) 1.0 else -1.0;
    const weight = 1.0 + std.math.log(f64, 2.0, @as(f64, @floatFromInt(count + 1))) * 0.1;
    out[idx] += sign * weight;
}

fn embeddingToJson(allocator: Allocator, vector: *const [EMBED_DIM]f64) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    const w = out.writer();
    try w.writeAll("[");
    for (vector.*, 0..) |v, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{d}", .{v});
    }
    try w.writeAll("]");
    return out.toOwnedSlice();
}

fn parseEmbedding(allocator: Allocator, json: []const u8) ![EMBED_DIM]f64 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    var out: [EMBED_DIM]f64 = undefined;
    for (&out) |*v| v.* = 0.0;
    switch (parsed.value) {
        .array => |arr| {
            const n = @min(arr.items.len, EMBED_DIM);
            var i: usize = 0;
            while (i < n) : (i += 1) {
                out[i] = switch (arr.items[i]) {
                    .integer => |iv| @floatFromInt(iv),
                    .float => |fv| fv,
                    else => 0.0,
                };
            }
        },
        else => return error.InvalidEmbedding,
    }
    return out;
}

fn cosine(a: *const [EMBED_DIM]f64, b: *const [EMBED_DIM]f64) f64 {
    var dot: f64 = 0.0;
    var na: f64 = 0.0;
    var nb: f64 = 0.0;
    var i: usize = 0;
    while (i < EMBED_DIM) : (i += 1) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na <= 0 or nb <= 0) return 0.0;
    return dot / (std.math.sqrt(na) * std.math.sqrt(nb));
}

fn makeFtsQuery(allocator: Allocator, text: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    var hash_set = std.StringHashMap(void).init(allocator);
    var token = std.ArrayList(u8).init(allocator);
    defer token.deinit();
    var emitted: usize = 0;
    for (text) |c0| {
        const c = std.ascii.toLower(c0);
        if (std.ascii.isAlphanumeric(c)) {
            if (token.items.len < 32) try token.append(c);
        } else {
            if (token.items.len >= 3 and emitted < 12 and !hash_set.contains(token.items)) {
                const copy = try allocator.dupe(u8, token.items);
                try hash_set.put(copy, {});
                if (emitted != 0) try out.writeAll(" OR ");
                try out.writer().print("{s}", .{token.items});
                emitted += 1;
            }
            token.clearRetainingCapacity();
        }
    }
    var it = hash_set.iterator();
    while (it.next()) |entry| allocator.free(entry.key_ptr.*);
    hash_set.deinit();
    return out.toOwnedSlice();
}

fn resolveWorkspacePath(allocator: Allocator, root: []const u8, rel: []const u8) ![]u8 {
    if (rel.len == 0) return error.InvalidPath;
    if (std.mem.startsWith(u8, rel, "/") or std.mem.indexOfScalar(u8, rel, '\\') != null or std.mem.indexOfScalar(u8, rel, 0) != null or std.mem.indexOfScalar(u8, rel, ':') != null) return error.UnauthorizedPath;
    var parts = std.mem.splitScalar(u8, rel, '/');
    while (parts.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return error.UnauthorizedPath;
    }
    try std.fs.cwd().makePath(root);
    const root_real = try std.fs.cwd().realpathAlloc(allocator, root);
    defer allocator.free(root_real);
    const full = try std.fs.path.join(allocator, &.{ root_real, rel });
    return full;
}

fn appendFileLineOriented(allocator: Allocator, path: []const u8, content: []const u8, unique: bool) !usize {
    const existing = std.fs.cwd().readFileAlloc(allocator, path, 64 * 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => try allocator.dupe(u8, ""),
        else => return err,
    };
    defer allocator.free(existing);
    var map = std.StringHashMap(void).init(allocator);
    defer map.deinit();
    if (unique) {
        var lines_existing = std.mem.splitScalar(u8, existing, '\n');
        while (lines_existing.next()) |line_raw| {
            const line = std.mem.trimRight(u8, line_raw, "\r");
            if (line.len > 0) try map.put(line, {});
        }
    }
    var out = std.ArrayList(u8).init(allocator);
    defer out.deinit();
    try out.appendSlice(existing);
    if (out.items.len > 0 and out.items[out.items.len - 1] != '\n') try out.append('\n');
    var appended: usize = 0;
    var lines_new = std.mem.splitScalar(u8, content, '\n');
    while (lines_new.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");
        if (line.len == 0) continue;
        if (unique and map.contains(line)) continue;
        try out.appendSlice(line);
        try out.append('\n');
        appended += 1;
    }
    try atomicWriteFile(allocator, path, out.items);
    return appended;
}

fn replaceLines(allocator: Allocator, path: []const u8, start_line: usize, end_line: usize, content: []const u8) !void {
    const existing = try std.fs.cwd().readFileAlloc(allocator, path, 64 * 1024 * 1024);
    defer allocator.free(existing);
    var lines = std.ArrayList([]const u8).init(allocator);
    defer lines.deinit();
    var it = std.mem.splitScalar(u8, existing, '\n');
    while (it.next()) |line| try lines.append(std.mem.trimRight(u8, line, "\r"));
    if (start_line == 0 or end_line < start_line or start_line > lines.items.len + 1) return error.InvalidLineRange;
    var out = std.ArrayList(u8).init(allocator);
    defer out.deinit();
    var i: usize = 1;
    while (i < start_line and i <= lines.items.len) : (i += 1) {
        try out.appendSlice(lines.items[i - 1]);
        try out.append('\n');
    }
    var new_lines = std.mem.splitScalar(u8, content, '\n');
    while (new_lines.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");
        try out.appendSlice(line);
        try out.append('\n');
    }
    i = end_line + 1;
    while (i <= lines.items.len) : (i += 1) {
        try out.appendSlice(lines.items[i - 1]);
        try out.append('\n');
    }
    try atomicWriteFile(allocator, path, out.items);
}

fn checkLines(allocator: Allocator, path: []const u8, lines_value: std.json.Value) ![]u8 {
    const existing = std.fs.cwd().readFileAlloc(allocator, path, 64 * 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => try allocator.dupe(u8, ""),
        else => return err,
    };
    defer allocator.free(existing);
    var map = std.StringHashMap(void).init(allocator);
    defer map.deinit();
    var existing_lines = std.mem.splitScalar(u8, existing, '\n');
    while (existing_lines.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");
        try map.put(line, {});
    }
    var out = std.ArrayList(u8).init(allocator);
    const w = out.writer();
    try w.writeAll("[");
    switch (lines_value) {
        .array => |arr| {
            for (arr.items, 0..) |item, i| {
                if (i != 0) try w.writeAll(",");
                switch (item) {
                    .string => |s| try w.print("{}", .{map.contains(s)}),
                    else => try w.writeAll("false"),
                }
            }
        },
        else => {},
    }
    try w.writeAll("]");
    return out.toOwnedSlice();
}

fn atomicWriteFile(allocator: Allocator, path: []const u8, data: []const u8) !void {
    const parent = std.fs.path.dirname(path);
    if (parent) |p| try std.fs.cwd().makePath(p);
    const tmp = try std.fmt.allocPrint(allocator, "{s}.tmp.{}", .{ path, nowMillis() });
    defer allocator.free(tmp);
    {
        var file = try std.fs.cwd().createFile(tmp, .{ .truncate = true });
        defer file.close();
        try file.writeAll(data);
        try file.sync();
    }
    std.fs.cwd().rename(tmp, path) catch |err| {
        std.fs.cwd().deleteFile(path) catch {};
        if (err != error.PathAlreadyExists) return err;
        try std.fs.cwd().rename(tmp, path);
    };
}

fn isHttpHostAllowed(allowed: []const u8, url: []const u8) bool {
    if (!std.mem.startsWith(u8, url, "https://") and !std.mem.startsWith(u8, url, "http://")) return false;
    if (std.mem.eql(u8, allowed, "*")) return true;
    const proto_end = std.mem.indexOf(u8, url, "://") orelse return false;
    const rest = url[proto_end + 3 ..];
    const host_end = std.mem.indexOfAny(u8, rest, "/:?#") orelse rest.len;
    const host = rest[0..host_end];
    if (host.len == 0) return false;
    var it = std.mem.splitScalar(u8, allowed, ',');
    while (it.next()) |item_raw| {
        const item = std.mem.trim(u8, item_raw, " \t\r\n");
        if (std.mem.eql(u8, item, "*") or std.ascii.eqlIgnoreCase(item, host)) return true;
    }
    return false;
}

fn joinUrl(allocator: Allocator, base: []const u8, suffix: []const u8) ![]u8 {
    if (std.mem.endsWith(u8, base, "/")) return std.fmt.allocPrint(allocator, "{s}{s}", .{ base, suffix });
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ base, suffix });
}
