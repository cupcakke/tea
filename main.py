from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unicodedata
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format=LOG_FORMAT, stream=sys.stdout)
LOG = logging.getLogger("agent.runtime")

ROOT_DIR = Path(os.environ.get("AGENT_ROOT", Path(__file__).resolve().parent)).resolve()
DATA_DIR = Path(os.environ.get("AGENT_DATA_DIR", ROOT_DIR / "agent_data")).resolve()
WORKSPACE_ROOT = Path(os.environ.get("AGENT_WORKSPACE", DATA_DIR / "workspaces")).resolve()
WIKI_ROOT = Path(os.environ.get("AGENT_WIKI", DATA_DIR / "wiki")).resolve()
SKILL_ROOT = Path(os.environ.get("AGENT_SKILLS", DATA_DIR / "skills")).resolve()
DB_PATH = Path(os.environ.get("AGENT_DB", DATA_DIR / "runtime.sqlite3")).resolve()
STATIC_INDEX = Path(os.environ.get("AGENT_INDEX_HTML", ROOT_DIR / "index.html")).resolve()

for _d in (DATA_DIR, WORKSPACE_ROOT, WIKI_ROOT, SKILL_ROOT):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_BASE_URL = os.environ.get("MODULAR_BASE_URL", "https://api.modular.com/v1")
MODEL_NAME = os.environ.get("MODULAR_MODEL", "zai-org/glm-5.3")
MODEL_API_KEY = os.environ.get("MODULAR_API_KEY", "")
MODEL_TEMPERATURE = float(os.environ.get("MODEL_TEMPERATURE", "0.96"))
MODEL_TOP_P = float(os.environ.get("MODEL_TOP_P", "1"))
MODEL_MAX_TOKENS = int(os.environ.get("MODEL_MAX_TOKENS", "100000"))
MODEL_FREQUENCY_PENALTY = float(os.environ.get("MODEL_FREQUENCY_PENALTY", "0.8"))
MODEL_PRESENCE_PENALTY = float(os.environ.get("MODEL_PRESENCE_PENALTY", "0.5"))
MODEL_SEED = int(os.environ.get("MODEL_SEED", "1234"))

SYSTEM2_HZ = float(os.environ.get("SYSTEM2_HZ", "1.0"))
SYSTEM1_HZ = float(os.environ.get("SYSTEM1_HZ", "20.0"))
COGNITION_K = int(os.environ.get("COGNITION_K", "8"))
COGNITION_H = int(os.environ.get("COGNITION_H", "32"))
MAX_STEP_RETRIES = int(os.environ.get("MAX_STEP_RETRIES", "3"))
DEFAULT_TOKEN_BUDGET = int(os.environ.get("DEFAULT_TOKEN_BUDGET", "2000000"))
DEFAULT_MAX_STEPS = int(os.environ.get("DEFAULT_MAX_STEPS", "160"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "1"))
ADMIN_TOKEN = os.environ.get("AGENT_ADMIN_TOKEN", "")
ALLOW_SHELL = os.environ.get("AGENT_ALLOW_SHELL", "1") not in ("0", "false", "False")
SHELL_TIMEOUT = int(os.environ.get("AGENT_SHELL_TIMEOUT", "60"))
MAX_FILE_BYTES = int(os.environ.get("AGENT_MAX_FILE_BYTES", str(8 * 1024 * 1024)))
EMBED_DIM = int(os.environ.get("AGENT_EMBED_DIM", "512"))
RRF_K = int(os.environ.get("AGENT_RRF_K", "60"))
GIT_ENABLED = shutil.which("git") is not None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime] = None) -> str:
    return (dt or utcnow()).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def jload(raw: Optional[str], default: Any = None) -> Any:
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 3.6))


TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with",
    "that", "this", "be", "as", "are", "was", "were", "by", "at", "from", "but", "not",
    "we", "you", "i", "he", "she", "they", "if", "then", "than", "so", "do", "does",
}


def tokenize(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFKD", (text or "").lower())
    return [t for t in TOKEN_RE.findall(normalized) if t not in _STOPWORDS and len(t) > 1]


def hashed_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    vec = [0.0] * dim
    toks = tokenize(text)
    if not toks:
        return vec
    counts: Dict[str, int] = defaultdict(int)
    for t in toks:
        counts[t] += 1
    for tok, cnt in counts.items():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign * (1.0 + math.log(cnt))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
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
    return dot / math.sqrt(na * nb)


def encode_vector(vec: Sequence[float]) -> str:
    return base64.b64encode(json.dumps([round(float(v), 6) for v in vec]).encode("utf-8")).decode("ascii")


def decode_vector(raw: Optional[str]) -> List[float]:
    if not raw:
        return []
    try:
        return list(json.loads(base64.b64decode(raw.encode("ascii")).decode("utf-8")))
    except Exception:
        return []


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, ident in enumerate(ranking):
            scores[ident] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def sinusoidal_staleness(seconds: float, dims: int = 8) -> List[float]:
    out: List[float] = []
    value = max(0.0, float(seconds))
    for i in range(dims // 2):
        freq = 1.0 / (10000.0 ** (2 * i / max(1, dims)))
        out.append(math.sin(value * freq))
        out.append(math.cos(value * freq))
    return [round(v, 6) for v in out[:dims]]


class RunState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


class NodeKind(str, Enum):
    PERCEIVE = "perceive"
    DELIBERATE = "deliberate"
    ACT = "act"
    VALIDATE = "validate"
    REFLECT = "reflect"
    CONSOLIDATE = "consolidate"
    TERMINAL = "terminal"


class ToolRisk(str, Enum):
    SAFE = "safe"
    GUARDED = "guarded"
    PRIVILEGED = "privileged"


DELETE_SENTINEL = "__DELETE__"


class SchemaError(ValueError):
    pass


class ValidationRejected(Exception):
    def __init__(self, reasons: List[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class BudgetExceeded(Exception):
    pass


class AuthorizationDenied(Exception):
    pass


@dataclass
class ProcedureSpec:
    spec_id: str
    tenant_id: str
    title: str
    objective: str
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    max_steps: int = DEFAULT_MAX_STEPS
    token_budget: int = DEFAULT_TOKEN_BUDGET
    verifier_program: Optional[str] = None
    created_at: str = field(default_factory=iso)

    def frozen_view(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "title": self.title,
            "objective": self.objective,
            "constraints": list(self.constraints),
            "success_criteria": list(self.success_criteria),
            "allowed_tools": sorted(set(self.allowed_tools)),
            "max_steps": self.max_steps,
        }

    def to_row(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "tenant_id": self.tenant_id,
            "title": self.title,
            "objective": self.objective,
            "constraints": jdump(self.constraints),
            "success_criteria": jdump(self.success_criteria),
            "allowed_tools": jdump(self.allowed_tools),
            "max_steps": self.max_steps,
            "token_budget": self.token_budget,
            "verifier_program": self.verifier_program or "",
            "created_at": self.created_at,
        }

    @staticmethod
    def from_row(row: sqlite3.Row) -> "ProcedureSpec":
        return ProcedureSpec(
            spec_id=row["spec_id"],
            tenant_id=row["tenant_id"],
            title=row["title"],
            objective=row["objective"],
            constraints=jload(row["constraints"], []) or [],
            success_criteria=jload(row["success_criteria"], []) or [],
            allowed_tools=jload(row["allowed_tools"], []) or [],
            max_steps=int(row["max_steps"]),
            token_budget=int(row["token_budget"]),
            verifier_program=row["verifier_program"] or None,
            created_at=row["created_at"],
        )


STATE_TOP_KEYS = {
    "phase",
    "progress",
    "subgoals",
    "facts",
    "artifacts",
    "blockers",
    "constraints_observed",
    "next_intent",
    "scratch",
    "metrics",
    "skill_hints",
}


@dataclass
class ExecutionState:
    phase: str = "bootstrap"
    progress: float = 0.0
    subgoals: List[Dict[str, Any]] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    blockers: List[str] = field(default_factory=list)
    constraints_observed: List[str] = field(default_factory=list)
    next_intent: str = ""
    scratch: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    skill_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "progress": round(float(self.progress), 4),
            "subgoals": self.subgoals,
            "facts": self.facts,
            "artifacts": self.artifacts,
            "blockers": self.blockers,
            "constraints_observed": self.constraints_observed,
            "next_intent": self.next_intent,
            "scratch": self.scratch,
            "metrics": self.metrics,
            "skill_hints": self.skill_hints,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ExecutionState":
        base = ExecutionState()
        if not isinstance(payload, dict):
            return base
        base.phase = str(payload.get("phase", base.phase))
        try:
            base.progress = clamp(float(payload.get("progress", 0.0)), 0.0, 1.0)
        except (TypeError, ValueError):
            base.progress = 0.0
        base.subgoals = payload.get("subgoals") if isinstance(payload.get("subgoals"), list) else []
        base.facts = payload.get("facts") if isinstance(payload.get("facts"), dict) else {}
        base.artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
        base.blockers = [str(b) for b in payload.get("blockers", []) if isinstance(payload.get("blockers"), list)]
        co = payload.get("constraints_observed")
        base.constraints_observed = [str(c) for c in co] if isinstance(co, list) else []
        base.next_intent = str(payload.get("next_intent", ""))
        base.scratch = payload.get("scratch") if isinstance(payload.get("scratch"), dict) else {}
        base.metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        sh = payload.get("skill_hints")
        base.skill_hints = [str(s) for s in sh] if isinstance(sh, list) else []
        return base

    def compact(self, max_chars: int = 6000) -> Dict[str, Any]:
        data = self.to_dict()
        data["subgoals"] = data["subgoals"][:24]
        data["blockers"] = data["blockers"][:12]
        data["skill_hints"] = data["skill_hints"][:8]
        text = jdump(data)
        if len(text) <= max_chars:
            return data
        data["facts"] = _truncate_mapping(data["facts"], max_chars // 3)
        data["artifacts"] = _truncate_mapping(data["artifacts"], max_chars // 4)
        data["scratch"] = _truncate_mapping(data["scratch"], max_chars // 6)
        return data


def _truncate_mapping(mapping: Dict[str, Any], budget: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    used = 0
    for key in sorted(mapping.keys()):
        chunk = jdump({key: mapping[key]})
        if used + len(chunk) > budget:
            out["__truncated__"] = True
            break
        out[key] = mapping[key]
        used += len(chunk)
    return out


@dataclass
class Observation:
    step: int
    source: str
    ok: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: int = 0
    created_at: str = field(default_factory=iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "source": self.source,
            "ok": self.ok,
            "payload": self.payload,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ActionCommand:
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    rationale_digest: str = ""
    terminal: bool = False
    final_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "rationale_digest": self.rationale_digest,
            "terminal": self.terminal,
            "final_answer": self.final_answer,
        }


@dataclass
class StepDecision:
    delta: Dict[str, Any]
    action: ActionCommand
    reasoning_tokens: int
    raw_len: int


@dataclass
class Skill:
    skill_id: str
    tenant_id: str
    name: str
    description: str
    trigger_signature: str
    procedure: List[str]
    tools: List[str]
    version: int = 1
    success_count: int = 0
    failure_count: int = 0
    quarantined: int = 0
    embedding: List[float] = field(default_factory=list)
    updated_at: str = field(default_factory=iso)

    @property
    def reliability(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return (self.success_count + 1.0) / (total + 2.0)

    def prompt_view(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "when_to_use": self.trigger_signature,
            "procedure": self.procedure[:12],
            "tools": self.tools[:12],
            "reliability": round(self.reliability, 3),
            "version": self.version,
        }

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Skill":
        return Skill(
            skill_id=row["skill_id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            description=row["description"],
            trigger_signature=row["trigger_signature"],
            procedure=jload(row["procedure"], []) or [],
            tools=jload(row["tools"], []) or [],
            version=int(row["version"]),
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
            quarantined=int(row["quarantined"]),
            embedding=decode_vector(row["embedding"]),
            updated_at=row["updated_at"],
        )


@dataclass
class ExecutionTrace:
    trace_id: str
    run_id: str
    step: int
    pre_state_digest: str
    selected_skill: Optional[str]
    tool: str
    outcome: str
    delta_digest: str
    post_state_digest: str
    receipt: Dict[str, Any]
    created_at: str = field(default_factory=iso)


@dataclass
class ReflectionPatch:
    patch_id: str
    run_id: str
    verdict: bool
    failure_point: str
    root_cause: str
    pivot_actions: List[str]
    memory_target: str
    guidance: str
    created_at: str = field(default_factory=iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "run_id": self.run_id,
            "verdict": self.verdict,
            "failure_point": self.failure_point,
            "root_cause": self.root_cause,
            "pivot_actions": self.pivot_actions,
            "memory_target": self.memory_target,
            "guidance": self.guidance,
        }


@dataclass
class CognitionFrame:
    frame_id: str
    run_id: str
    generated_at: float
    tokens: List[List[float]]
    gates: Dict[str, float]
    subgoal: str
    directive: str
    horizon: int

    def staleness(self, now: Optional[float] = None) -> float:
        return max(0.0, (now if now is not None else time.time()) - self.generated_at)

    def prompt_view(self, now: Optional[float] = None) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "subgoal": self.subgoal,
            "directive": self.directive,
            "gates": {k: round(v, 4) for k, v in self.gates.items()},
            "horizon": self.horizon,
            "staleness_s": round(self.staleness(now), 3),
            "staleness_encoding": sinusoidal_staleness(self.staleness(now)),
            "cognition_digest": [round(sum(t) / max(1, len(t)), 5) for t in self.tokens[:COGNITION_K]],
        }


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=15000;

CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    api_key_hash TEXT NOT NULL UNIQUE,
    token_budget INTEGER NOT NULL DEFAULT 2000000,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    allowed_risk TEXT NOT NULL DEFAULT 'guarded',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS specs (
    spec_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    constraints TEXT NOT NULL,
    success_criteria TEXT NOT NULL,
    allowed_tools TEXT NOT NULL,
    max_steps INTEGER NOT NULL,
    token_budget INTEGER NOT NULL,
    verifier_program TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_specs_tenant ON specs(tenant_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    spec_id TEXT NOT NULL REFERENCES specs(spec_id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    node TEXT NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    retries INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    last_observation TEXT NOT NULL DEFAULT '{}',
    final_answer TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    lease_owner TEXT NOT NULL DEFAULT '',
    lease_expires_at REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_tenant ON runs(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_conv ON runs(conversation_id);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    node TEXT NOT NULL,
    state TEXT NOT NULL,
    observation TEXT NOT NULL,
    status TEXT NOT NULL,
    tokens_used INTEGER NOT NULL,
    digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ckpt_run ON checkpoints(run_id, step);

CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    pre_state_digest TEXT NOT NULL,
    selected_skill TEXT,
    tool TEXT NOT NULL,
    outcome TEXT NOT NULL,
    delta_digest TEXT NOT NULL,
    post_state_digest TEXT NOT NULL,
    receipt TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_run ON traces(run_id, step);
CREATE INDEX IF NOT EXISTS idx_traces_outcome ON traces(outcome);

CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    trigger_signature TEXT NOT NULL,
    procedure TEXT NOT NULL,
    tools TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    quarantined INTEGER NOT NULL DEFAULT 0,
    embedding TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_skills_unique ON skills(tenant_id, name);
CREATE INDEX IF NOT EXISTS idx_skills_tenant ON skills(tenant_id, quarantined);

CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    skill_id UNINDEXED,
    tenant_id UNINDEXED,
    text,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS skill_versions (
    version_id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    snapshot TEXT NOT NULL,
    reason TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skillver ON skill_versions(skill_id, version);

CREATE TABLE IF NOT EXISTS reflections (
    patch_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    verdict INTEGER NOT NULL,
    failure_point TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    pivot_actions TEXT NOT NULL,
    memory_target TEXT NOT NULL,
    guidance TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reflections_run ON reflections(run_id);

CREATE TABLE IF NOT EXISTS distillation_samples (
    sample_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    student_prompt_digest TEXT NOT NULL,
    teacher_prompt_digest TEXT NOT NULL,
    tokens TEXT NOT NULL,
    student_logprobs TEXT NOT NULL,
    teacher_logprobs TEXT NOT NULL,
    reverse_kl REAL NOT NULL,
    weight REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_distill_run ON distillation_samples(run_id);

CREATE TABLE IF NOT EXISTS policy_priors (
    prior_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    signature TEXT NOT NULL,
    directive TEXT NOT NULL,
    logit REAL NOT NULL,
    updates INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_prior_unique ON policy_priors(tenant_id, signature);

CREATE TABLE IF NOT EXISTS wiki_pages (
    page_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    embedding TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_wiki_unique ON wiki_pages(tenant_id, slug);

CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
    page_id UNINDEXED,
    tenant_id UNINDEXED,
    text,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS raw_events (
    event_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON raw_events(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_kind ON raw_events(kind);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_tenant ON conversations(tenant_id, updated_at);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS diagnostic_tasks (
    task_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    name TEXT NOT NULL,
    payload TEXT NOT NULL,
    expectation TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_diag_tenant ON diagnostic_tasks(tenant_id);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    resource TEXT NOT NULL,
    allowed INTEGER NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, created_at);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=15000")
            self._local.conn = conn
        return conn

    @contextlib.contextmanager
    def connect(self) -> Iterable[sqlite3.Connection]:
        conn = self._conn()
        yield conn

    @contextlib.contextmanager
    def tx(self) -> Iterable[sqlite3.Connection]:
        with self._write_lock:
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        conn = self._conn()
        cur = conn.execute(sql, tuple(params))
        rows = cur.fetchall()
        cur.close()
        return rows

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self.tx() as conn:
            conn.execute(sql, tuple(params))

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> None:
        with self.tx() as conn:
            conn.executemany(sql, [tuple(s) for s in seq])


DB = Database(DB_PATH)


class Ledger:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = threading.RLock()

    def append(self, tenant_id: str, kind: str, payload: Dict[str, Any], run_id: str = "") -> str:
        with self._lock:
            last = self.db.query_one("SELECT hash FROM raw_events ORDER BY rowid DESC LIMIT 1")
            prev_hash = last["hash"] if last else "0" * 64
            event_id = new_id("evt")
            created = iso()
            body = {
                "event_id": event_id,
                "tenant_id": tenant_id,
                "run_id": run_id,
                "kind": kind,
                "payload": payload,
                "prev_hash": prev_hash,
                "created_at": created,
            }
            digest = stable_hash(body)
            self.db.execute(
                "INSERT INTO raw_events(event_id, tenant_id, run_id, kind, payload, prev_hash, hash, created_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (event_id, tenant_id, run_id, kind, jdump(payload), prev_hash, digest, created),
            )
            return event_id

    def verify_chain(self, limit: int = 5000) -> Dict[str, Any]:
        rows = self.db.query("SELECT * FROM raw_events ORDER BY rowid ASC LIMIT ?", (limit,))
        prev = "0" * 64
        broken: List[str] = []
        for row in rows:
            body = {
                "event_id": row["event_id"],
                "tenant_id": row["tenant_id"],
                "run_id": row["run_id"],
                "kind": row["kind"],
                "payload": jload(row["payload"], {}),
                "prev_hash": row["prev_hash"],
                "created_at": row["created_at"],
            }
            if row["prev_hash"] != prev or stable_hash(body) != row["hash"]:
                broken.append(row["event_id"])
            prev = row["hash"]
        return {"checked": len(rows), "broken": broken, "intact": not broken}


LEDGER = Ledger(DB)


def hash_key(raw: str) -> str:
    return hashlib.sha256(("agent-runtime-v1:" + raw).encode("utf-8")).hexdigest()


@dataclass
class Tenant:
    tenant_id: str
    name: str
    token_budget: int
    tokens_used: int
    allowed_risk: ToolRisk

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Tenant":
        try:
            risk = ToolRisk(row["allowed_risk"])
        except ValueError:
            risk = ToolRisk.GUARDED
        return Tenant(
            tenant_id=row["tenant_id"],
            name=row["name"],
            token_budget=int(row["token_budget"]),
            tokens_used=int(row["tokens_used"]),
            allowed_risk=risk,
        )


class TenantRegistry:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = threading.RLock()
        self._ensure_default()

    def _ensure_default(self) -> None:
        row = self.db.query_one("SELECT COUNT(*) AS c FROM tenants")
        if row and int(row["c"]) > 0:
            return
        key = os.environ.get("AGENT_DEFAULT_API_KEY", "local-dev-key")
        self.create("default", key, DEFAULT_TOKEN_BUDGET, ToolRisk.PRIVILEGED)
        LOG.info("Bootstrapped default tenant with api key: %s", key)

    def create(self, name: str, api_key: str, budget: int, risk: ToolRisk) -> Tenant:
        with self._lock:
            tenant_id = new_id("tnt")
            self.db.execute(
                "INSERT INTO tenants(tenant_id, name, api_key_hash, token_budget, tokens_used, allowed_risk, created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (tenant_id, name, hash_key(api_key), budget, 0, risk.value, iso()),
            )
            WORKSPACE_ROOT.joinpath(tenant_id).mkdir(parents=True, exist_ok=True)
            WIKI_ROOT.joinpath(tenant_id).mkdir(parents=True, exist_ok=True)
            SKILL_ROOT.joinpath(tenant_id).mkdir(parents=True, exist_ok=True)
            return Tenant(tenant_id, name, budget, 0, risk)

    def by_api_key(self, api_key: str) -> Optional[Tenant]:
        row = self.db.query_one("SELECT * FROM tenants WHERE api_key_hash = ?", (hash_key(api_key),))
        return Tenant.from_row(row) if row else None

    def by_id(self, tenant_id: str) -> Optional[Tenant]:
        row = self.db.query_one("SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,))
        return Tenant.from_row(row) if row else None

    def charge(self, tenant_id: str, tokens: int) -> None:
        with self._lock:
            row = self.db.query_one("SELECT token_budget, tokens_used FROM tenants WHERE tenant_id=?", (tenant_id,))
            if not row:
                raise AuthorizationDenied("unknown tenant")
            used = int(row["tokens_used"]) + max(0, tokens)
            if used > int(row["token_budget"]):
                self.db.execute("UPDATE tenants SET tokens_used=? WHERE tenant_id=?", (used, tenant_id))
                raise BudgetExceeded(f"tenant token budget exhausted ({used}/{row['token_budget']})")
            self.db.execute("UPDATE tenants SET tokens_used=? WHERE tenant_id=?", (used, tenant_id))

    def list_all(self) -> List[Tenant]:
        return [Tenant.from_row(r) for r in self.db.query("SELECT * FROM tenants ORDER BY created_at")]


TENANTS = TenantRegistry(DB)


class ModelClient:
    def __init__(self) -> None:
        self._client: Optional[OpenAI] = None
        self._lock = threading.RLock()
        self.available = bool(MODEL_API_KEY)

    def client(self) -> OpenAI:
        with self._lock:
            if self._client is None:
                if not MODEL_API_KEY:
                    raise RuntimeError("MODULAR_API_KEY is not configured")
                self._client = OpenAI(base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)
            return self._client

    def _params(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": MODEL_NAME,
            "temperature": MODEL_TEMPERATURE,
            "top_p": MODEL_TOP_P,
            "max_tokens": MODEL_MAX_TOKENS,
            "frequency_penalty": MODEL_FREQUENCY_PENALTY,
            "presence_penalty": MODEL_PRESENCE_PENALTY,
            "seed": MODEL_SEED,
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        return params

    def complete(self, messages: List[Dict[str, str]], **overrides: Any) -> Tuple[str, int]:
        params = self._params(overrides)
        client = self.client()
        text_parts: List[str] = []
        usage_tokens = 0
        stream = client.chat.completions.create(
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **params,
        )
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                total = getattr(usage, "total_tokens", None)
                if isinstance(total, int):
                    usage_tokens = total
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                text_parts.append(piece)
        text = "".join(text_parts)
        if usage_tokens <= 0:
            usage_tokens = approx_tokens(" ".join(m.get("content", "") for m in messages)) + approx_tokens(text)
        return text, usage_tokens

    def stream(self, messages: List[Dict[str, str]], **overrides: Any) -> Iterable[Tuple[str, Optional[int]]]:
        params = self._params(overrides)
        client = self.client()
        stream = client.chat.completions.create(
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **params,
        )
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            total: Optional[int] = None
            if usage is not None:
                candidate = getattr(usage, "total_tokens", None)
                if isinstance(candidate, int):
                    total = candidate
            if not getattr(chunk, "choices", None):
                if total is not None:
                    yield "", total
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece, total
            elif total is not None:
                yield "", total


MODEL = ModelClient()


class GrammarDecoder:
    OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
    FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

    @classmethod
    def extract_json(cls, raw: str) -> Dict[str, Any]:
        if not raw or not raw.strip():
            raise SchemaError("empty model response")
        candidates: List[str] = []
        for match in cls.FENCE_RE.finditer(raw):
            candidates.append(match.group(1))
        candidates.append(raw)
        obj_match = cls.OBJECT_RE.search(raw)
        if obj_match:
            candidates.append(obj_match.group(0))
        for cand in candidates:
            cand = cand.strip()
            if not cand:
                continue
            for attempt in (cand, cls._repair(cand)):
                try:
                    parsed = json.loads(attempt)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
        balanced = cls._balanced_scan(raw)
        if balanced is not None:
            return balanced
        raise SchemaError("model response did not contain a decodable JSON object")

    @staticmethod
    def _repair(text: str) -> str:
        cleaned = re.sub(r",\s*(\}|\])", r"\1", text)
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        cleaned = re.sub(r"//[^\n\r]*", "", cleaned)
        opens = cleaned.count("{") - cleaned.count("}")
        if opens > 0:
            cleaned = cleaned + ("}" * opens)
        brackets = cleaned.count("[") - cleaned.count("]")
        if brackets > 0:
            cleaned = cleaned + ("]" * brackets)
        return cleaned

    @staticmethod
    def _balanced_scan(raw: str) -> Optional[Dict[str, Any]]:
        start = raw.find("{")
        while start != -1:
            depth = 0
            in_str = False
            escape = False
            for idx in range(start, len(raw)):
                ch = raw[idx]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        blob = raw[start : idx + 1]
                        try:
                            parsed = json.loads(blob)
                        except json.JSONDecodeError:
                            try:
                                parsed = json.loads(GrammarDecoder._repair(blob))
                            except json.JSONDecodeError:
                                break
                        if isinstance(parsed, dict):
                            return parsed
                        break
            start = raw.find("{", start + 1)
        return None


class DeltaValidator
    MAX_DELTA_BYTES = 120_000
    MAX_LIST_LEN = 256
    MAX_DEPTH = 8

    @classmethod
    def validate(cls, delta: Any, allowed_tools: Set[str]) -> Dict[str, Any]:
        reasons: List[str] = []
        if not isinstance(delta, dict):
            raise ValidationRejected(["state delta must be a JSON object"])
        blob = jdump(delta)
        if len(blob) > cls.MAX_DELTA_BYTES:
            reasons.append(f"delta too large ({len(blob)} bytes)")
        unknown = [k for k in delta.keys() if k not in STATE_TOP_KEYS]
        if unknown:
            reasons.append(f"unknown state keys: {sorted(unknown)}")
        if "progress" in delta and delta["progress"] is not None:
            try:
                float(delta["progress"])
            except (TypeError, ValueError):
                reasons.append("progress must be numeric or null")
        for list_key in ("subgoals", "blockers", "constraints_observed", "skill_hints"):
            if list_key in delta and delta[list_key] is not None and not isinstance(delta[list_key], list):
                reasons.append(f"{list_key} must be a list or null")
            elif isinstance(delta.get(list_key), list) and len(delta[list_key]) > cls.MAX_LIST_LEN:
                reasons.append(f"{list_key} exceeds {cls.MAX_LIST_LEN} entries")
        for dict_key in ("facts", "artifacts", "scratch", "metrics"):
            if dict_key in delta and delta[dict_key] is not None and not isinstance(delta[dict_key], dict):
                reasons.append(f"{dict_key} must be an object or null")
        if cls._depth(delta) > cls.MAX_DEPTH:
            reasons.append("delta nesting too deep")
        if reasons:
            raise ValidationRejected(reasons)
        return delta

    @classmethod
    def _depth(cls, node: Any, level: int = 0) -> int:
        if level > cls.MAX_DEPTH + 2:
            return level
        if isinstance(node, dict):
            if not node:
                return level
            return max(cls._depth(v, level + 1) for v in node.values())
        if isinstance(node, list):
            if not node:
                return level
            return max(cls._depth(v, level + 1) for v in node)
        return level

    @staticmethod
    def merge(state: ExecutionState, delta: Dict[str, Any]) -> ExecutionState:
        current = state.to_dict()
        merged = DeltaValidator._merge_dict(current, delta)
        return ExecutionState.from_dict(merged)

    @staticmethod
    def _merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for key, value in patch.items():
            if value is None or value == DELETE_SENTINEL:
                out.pop(key, None)
                continue
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = DeltaValidator._merge_dict(out[key], value)
            else:
                out[key] = value
        return out

    @staticmethod
    def validate_action(payload: Any, allowed_tools: Set[str]) -> ActionCommand:
        reasons: List[str] = []
        if not isinstance(payload, dict):
            raise ValidationRejected(["action must be a JSON object"])
        tool = payload.get("tool")
        terminal = bool(payload.get("terminal", False))
        final_answer = payload.get("final_answer")
        if terminal:
            tool = tool or "finish"
        if not isinstance(tool, str) or not tool.strip():
            reasons.append("action.tool must be a non-empty string")
            tool = "noop"
        tool = tool.strip()
        if tool not in allowed_tools and tool not in ("finish", "noop"):
            reasons.append(f"tool '{tool}' is not authorized for this run")
        args = payload.get("arguments", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            reasons.append("action.arguments must be an object")
            args = {}
        if terminal and not isinstance(final_answer, str):
            final_answer = jdump(final_answer) if final_answer is not None else ""
        digest = payload.get("rationale_digest", "")
        if not isinstance(digest, str):
            digest = str(digest)
        if reasons:
            raise ValidationRejected(reasons)
        return ActionCommand(
            tool=tool,
            arguments=args,
            rationale_digest=digest[:400],
            terminal=terminal,
            final_answer=final_answer if isinstance(final_answer, str) else None,
        )


class OutputClassifier
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
        re.compile(r"disregard\s+the\s+system\s+prompt", re.I),
        re.compile(r"reveal\s+your\s+(system\s+)?prompt", re.I),
        re.compile(r"\bBEGIN\s+RSA\s+PRIVATE\s+KEY\b", re.I),
    ]
    SECRET_PATTERNS = [
        re.compile(r"sk-[A-Za-z0-9]{16,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ]

    @classmethod
    def classify(cls, text: str) -> Dict[str, Any]:
        flags: List[str] = []
        if not isinstance(text, str):
            text = str(text)
        for pat in cls.INJECTION_PATTERNS:
            if pat.search(text):
                flags.append("prompt_injection_echo")
                break
        for pat in cls.SECRET_PATTERNS:
            if pat.search(text):
                flags.append("secret_leak")
                break
        if len(text) > 400_000:
            flags.append("oversize_output")
        control = sum(1 for ch in text[:20000] if ord(ch) < 9 or (13 < ord(ch) < 32))
        if control > 32:
            flags.append("control_char_spam")
        return {"allowed": not flags, "flags": flags}

    @classmethod
    def redact(cls, text: str) -> str:
        out = text
        for pat in cls.SECRET_PATTERNS:
            out = pat.sub("[REDACTED_SECRET]", out)
        return out


class SecurityFence:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._risk_order = {ToolRisk.SAFE: 0, ToolRisk.GUARDED: 1, ToolRisk.PRIVILEGED: 2}

    def authorize(self, tenant: Tenant, tool_name: str, risk: ToolRisk, spec: ProcedureSpec, resource: str) -> None:
        allowed = True
        detail = ""
        if self._risk_order[risk] > self._risk_order[tenant.allowed_risk]:
            allowed = False
            detail = f"risk {risk.value} exceeds tenant ceiling {tenant.allowed_risk.value}"
        elif spec.allowed_tools and tool_name not in spec.allowed_tools and tool_name not in ("finish", "noop"):
            allowed = False
            detail = "tool not present in spec allow-list"
        self.audit(tenant.tenant_id, "agent", f"tool:{tool_name}", resource, allowed, detail)
        if not allowed:
            raise AuthorizationDenied(detail)

    def audit(self, tenant_id: str, actor: str, action: str, resource: str, allowed: bool, detail: str) -> None:
        self.db.execute(
            "INSERT INTO audit_log(audit_id, tenant_id, actor, action, resource, allowed, detail, created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (new_id("aud"), tenant_id, actor, action, resource, 1 if allowed else 0, detail[:2000], iso()),
        )


FENCE = SecurityFence(DB)


class FileSystemSandbox:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def resolve(self, relative: str) -> Path:
        if relative is None:
            raise ValueError("path required")
        candidate = str(relative).strip()
        if not candidate:
            raise ValueError("path required")
        if candidate.startswith("/"):
            candidate = candidate.lstrip("/")
        if "\x00" in candidate:
            raise ValueError("invalid path")
        target = (self.root / candidate).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("path escapes sandbox root")
        return target

    def write_file(self, path: str, content: str, mode: str = "overwrite") -> Dict[str, Any]:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content if isinstance(content, str) else jdump(content)
        if len(data.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError("content exceeds max file size")
        with self._lock:
            if mode == "append":
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(data)
            else:
                tmp = target.with_name(target.name + f".tmp.{secrets.token_hex(6)}")
                tmp.write_text(data, encoding="utf-8")
                os.replace(tmp, target)
        return {"path": str(target.relative_to(self.root)), "bytes": target.stat().st_size, "mode": mode}

    def read_file(self, path: str, start_line: int = 1, end_line: int = 0) -> Dict[str, Any]:
        target = self.resolve(path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        s = max(1, int(start_line))
        e = len(lines) if not end_line else min(len(lines), int(end_line))
        segment = lines[s - 1 : e] if s <= len(lines) else []
        body = "\n".join(segment)
        if len(body) > 200_000:
            body = body[:200_000] + "\n[TRUNCATED]"
        return {
            "path": str(target.relative_to(self.root)),
            "total_lines": len(lines),
            "start_line": s,
            "end_line": e,
            "content": body,
        }

    def append_file(self, path: str, lines: Sequence[str], unique: bool = True) -> Dict[str, Any]:
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        incoming = [str(l).rstrip("\n") for l in lines if str(l).strip() != ""]
        with self._lock:
            existing_text = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
            existing = existing_text.splitlines()
            existing_set = set(existing)
            added: List[str] = []
            skipped: List[str] = []
            for line in incoming:
                if unique and (line in existing_set or line in added):
                    skipped.append(line)
                    continue
                added.append(line)
            if added:
                needs_nl = bool(existing_text) and not existing_text.endswith("\n")
                with target.open("a", encoding="utf-8") as fh:
                    if needs_nl:
                        fh.write("\n")
                    fh.write("\n".join(added) + "\n")
        return {
            "path": str(target.relative_to(self.root)),
            "added": added,
            "skipped": skipped,
            "added_count": len(added),
            "skipped_count": len(skipped),
        }

    def replace_lines(self, path: str, start_line: int, end_line: int, replacement: Union[str, Sequence[str]]) -> Dict[str, Any]:
        target = self.resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"no such file: {path}")
        new_lines = replacement.splitlines() if isinstance(replacement, str) else [str(x) for x in replacement]
        with self._lock:
            original = target.read_text(encoding="utf-8", errors="replace").splitlines()
            s = max(1, int(start_line))
            e = min(len(original), int(end_line)) if int(end_line) > 0 else s - 1
            if s > len(original) + 1:
                raise ValueError("start_line beyond end of file")
            head = original[: s - 1]
            tail = original[e:] if e >= s - 1 else original[s - 1 :]
            merged = head + new_lines + tail
            tmp = target.with_name(target.name + f".tmp.{secrets.token_hex(6)}")
            tmp.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
            os.replace(tmp, target)
        return {
            "path": str(target.relative_to(self.root)),
            "replaced_range": [s, e],
            "removed_lines": max(0, e - s + 1),
            "inserted_lines": len(new_lines),
            "total_lines": len(merged),
        }

    def check_lines(self, path: str, lines: Sequence[str]) -> Dict[str, Any]:
        target = self.resolve(path)
        present: Dict[str, bool] = {}
        existing: Set[str] = set()
        if target.exists() and target.is_file():
            existing = set(target.read_text(encoding="utf-8", errors="replace").splitlines())
        for line in lines:
            key = str(line).rstrip("\n")
            present[key] = key in existing
        return {
            "path": str(target.relative_to(self.root)) if target.exists() else str(path),
            "results": present,
            "all_present": all(present.values()) if present else True,
            "missing": [k for k, v in present.items() if not v],
        }

    def list_dir(self, path: str = ".", depth: int = 1) -> Dict[str, Any]:
        target = self.resolve(path) if path not in ("", ".", "./") else self.root
        if not target.exists():
            raise FileNotFoundError(f"no such directory: {path}")
        entries: List[Dict[str, Any]] = []
        base_depth = len(target.parts)
        for item in sorted(target.rglob("*")):
            if len(item.parts) - base_depth > max(1, depth):
                continue
            try:
                rel = str(item.relative_to(self.root))
            except ValueError:
                continue
            entries.append(
                {
                    "path": rel,
                    "type": "dir" if item.is_dir() else "file",
                    "bytes": item.stat().st_size if item.is_file() else 0,
                }
            )
            if len(entries) >= 800:
                break
        return {"root": str(target.relative_to(self.root)) if target != self.root else ".", "entries": entries}

    def delete(self, path: str) -> Dict[str, Any]:
        target = self.resolve(path)
        if target == self.root:
            raise ValueError("cannot delete sandbox root")
        with self._lock:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            else:
                raise FileNotFoundError(f"no such path: {path}")
        return {"deleted": str(path)}

    def search(self, pattern: str, glob: str = "**/*") -> Dict[str, Any]:
        rx = re.compile(pattern)
        hits: List[Dict[str, Any]] = []
        for item in sorted(self.root.glob(glob)):
            if not item.is_file():
                continue
            try:
                text = item.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append({"path": str(item.relative_to(self.root)), "line": idx, "text": line[:400]})
                    if len(hits) >= 300:
                        return {"pattern": pattern, "hits": hits, "truncated": True}
        return {"pattern": pattern, "hits": hits, "truncated": False}


@dataclass
class ToolSpec:
    name: str
    description: str
    risk: ToolRisk
    parameters: Dict[str, Any]
    handler: Callable[["ToolContext", Dict[str, Any]], Dict[str, Any]]

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk": self.risk.value,
            "parameters": self.parameters,
        }


@dataclass
class ToolContext:
    tenant: Tenant
    spec: ProcedureSpec
    run_id: str
    step: int
    sandbox: FileSystemSandbox
    memory: "MemorySubsystem"
    state: ExecutionState


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def catalog(self, allowed: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        out = []
        for name in self.names():
            if allowed is not None and name not in allowed:
                continue
            out.append(self._tools[name].schema())
        return out

    def execute(self, ctx: ToolContext, action: ActionCommand) -> Observation:
        started = time.time()
        tool = self.get(action.tool)
        if tool is None:
            return Observation(
                step=ctx.step,
                source=action.tool,
                ok=False,
                error=f"unknown tool '{action.tool}'",
                latency_ms=int((time.time() - started) * 1000),
            )
        try:
            FENCE.authorize(ctx.tenant, tool.name, tool.risk, ctx.spec, f"run:{ctx.run_id}")
            payload = tool.handler(ctx, dict(action.arguments or {}))
            verdict = OutputClassifier.classify(jdump(payload))
            if not verdict["allowed"]:
                return Observation(
                    step=ctx.step,
                    source=tool.name,
                    ok=False,
                    error=f"output rejected by classifier: {verdict['flags']}",
                    latency_ms=int((time.time() - started) * 1000),
                )
            return Observation(
                step=ctx.step,
                source=tool.name,
                ok=True,
                payload=payload if isinstance(payload, dict) else {"result": payload},
                latency_ms=int((time.time() - started) * 1000),
            )
        except AuthorizationDenied as exc:
            return Observation(step=ctx.step, source=tool.name, ok=False, error=f"authorization denied: {exc}",
                               latency_ms=int((time.time() - started) * 1000))
        except Exception as exc:
            return Observation(
                step=ctx.step,
                source=tool.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.time() - started) * 1000),
            )


TOOLS = ToolRegistry()


def _tool_write_file(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return ctx.sandbox.write_file(args.get("path", ""), args.get("content", ""), str(args.get("mode", "overwrite")))


def _tool_read_file(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return ctx.sandbox.read_file(args.get("path", ""), int(args.get("start_line", 1)), int(args.get("end_line", 0)))


def _tool_append_file(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    lines = args.get("lines", [])
    if isinstance(lines, str):
        lines = lines.splitlines()
    return ctx.sandbox.append_file(args.get("path", ""), lines, bool(args.get("unique", True)))


def _tool_replace_lines(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return ctx.sandbox.replace_lines(
        args.get("path", ""),
        int(args.get("start_line", 1)),
        int(args.get("end_line", 0)),
        args.get("replacement", ""),
    )


def _tool_check_lines(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    lines = args.get("lines", [])
    if isinstance(lines, str):
        lines = lines.splitlines()
    return ctx.sandbox.check_lines(args.get("path", ""), lines)


def _tool_list_dir(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return ctx.sandbox.list_dir(str(args.get("path", ".")), int(args.get("depth", 1)))


def _tool_delete_path(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return ctx.sandbox.delete(args.get("path", ""))


def _tool_search_files(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return ctx.sandbox.search(str(args.get("pattern", "")), str(args.get("glob", "**/*")))


def _tool_run_python(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    code = args.get("code", "")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code required")
    workdir = ctx.sandbox.root
    script = workdir / f".exec_{secrets.token_hex(8)}.py"
    script.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(script)],
            capture_output=True,
            text=True,
            timeout=int(args.get("timeout", SHELL_TIMEOUT)),
            cwd=str(workdir),
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(workdir), "PYTHONIOENCODING": "utf-8"},
        )
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-40000:],
            "stderr": proc.stderr[-20000:],
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": 124, "stdout": "", "stderr": "execution timed out"}
    finally:
        with contextlib.suppress(OSError):
            script.unlink()


_SHELL_DENY = re.compile(
    r"(rm\s+-rf\s+/|:\(\)\{|mkfs|dd\s+if=|shutdown|reboot|curl\s+[^|]*\|\s*sh|wget\s+[^|]*\|\s*sh|chmod\s+777\s+/)",
    re.I,
)


def _tool_run_shell(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if not ALLOW_SHELL:
        raise PermissionError("shell execution disabled by configuration")
    command = args.get("command", "")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command required")
    if _SHELL_DENY.search(command):
        raise PermissionError("command blocked by deterministic guardrail")
    proc = subprocess.run(
        ["/bin/sh", "-c", command],
        capture_output=True,
        text=True,
        timeout=int(args.get("timeout", SHELL_TIMEOUT)),
        cwd=str(ctx.sandbox.root),
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(ctx.sandbox.root)},
    )
    return {"exit_code": proc.returncode, "stdout": proc.stdout[-40000:], "stderr": proc.stderr[-20000:]}


def _tool_memory_search(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    query = str(args.get("query", ""))
    limit = int(args.get("limit", 5))
    skills = ctx.memory.em.retrieve(ctx.tenant.tenant_id, query, limit=limit)
    pages = ctx.memory.wiki.search(ctx.tenant.tenant_id, query, limit=limit)
    return {
        "skills": [s.prompt_view() for s in skills],
        "wiki": [{"slug": p["slug"], "title": p["title"], "excerpt": p["body"][:900]} for p in pages],
    }


def _tool_memory_write(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    slug = str(args.get("slug", "")).strip()
    title = str(args.get("title", slug or "note")).strip()
    body = str(args.get("body", ""))
    if not slug:
        raise ValueError("slug required")
    page = ctx.memory.wiki.upsert(ctx.tenant.tenant_id, slug, title, body, reason=f"run:{ctx.run_id}")
    return {"slug": page["slug"], "revision": page["revision"]}


def _tool_record_skill(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    name = str(args.get("name", "")).strip()
    if not name:
        raise ValueError("skill name required")
    procedure = args.get("procedure", [])
    if isinstance(procedure, str):
        procedure = [ln for ln in procedure.splitlines() if ln.strip()]
    skill = ctx.memory.em.upsert(
        tenant_id=ctx.tenant.tenant_id,
        name=name,
        description=str(args.get("description", "")),
        trigger_signature=str(args.get("when_to_use", "")),
        procedure=[str(p) for p in procedure][:32],
        tools=[str(t) for t in (args.get("tools", []) or [])][:16],
        reason=f"agent-authored during {ctx.run_id}",
    )
    return {"skill_id": skill.skill_id, "version": skill.version, "name": skill.name}


def _tool_http_fetch(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    url = str(args.get("url", "")).strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("only http/https URLs are permitted")
    lowered = url.lower()
    for blocked in ("localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254", "[::1]", "metadata.google"):
        if blocked in lowered:
            raise PermissionError("target host blocked by SSRF guardrail")
    method = str(args.get("method", "GET")).upper()
    if method not in ("GET", "POST", "HEAD"):
        raise ValueError("unsupported method")
    data = args.get("body")
    encoded = jdump(data).encode("utf-8") if isinstance(data, (dict, list)) else (
        str(data).encode("utf-8") if data is not None else None
    )
    headers = {"User-Agent": "AutonomousAgentRuntime/1.0", "Accept": "*/*"}
    extra = args.get("headers")
    if isinstance(extra, dict):
        for k, v in list(extra.items())[:16]:
            if str(k).lower() not in ("host", "authorization", "cookie"):
                headers[str(k)] = str(v)
    req = urllib.request.Request(url, data=encoded, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=int(args.get("timeout", 25))) as resp:
            raw = resp.read(2_000_000)
            charset = resp.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")
            return {
                "status": resp.status,
                "url": resp.geturl(),
                "content_type": resp.headers.get("Content-Type", ""),
                "body": OutputClassifier.redact(text[:200_000]),
            }
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "url": url, "error": str(exc), "body": exc.read(50000).decode("utf-8", "replace")}
    except urllib.error.URLError as exc:
        raise RuntimeError(f"network error: {exc.reason}") from exc


def _tool_reason(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    question = str(args.get("question", "")).strip()
    if not question:
        raise ValueError("question required")
    messages = [
        {
            "role": "system",
            "content": "You are a precise analytical subroutine. Answer in plain text, no markdown. Be concise and factual.",
        },
        {
            "role": "user",
            "content": jdump(
                {
                    "task": ctx.spec.objective,
                    "state_digest": ctx.state.compact(2500),
                    "question": question,
                }
            ),
        },
    ]
    text, tokens = MODEL.complete(messages, max_tokens=min(4096, MODEL_MAX_TOKENS), temperature=0.4)
    TENANTS.charge(ctx.tenant.tenant_id, tokens)
    return {"answer": text.strip()[:20000], "tokens": tokens}


def _tool_finish(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return {"final_answer": str(args.get("final_answer", ""))[:100_000], "terminal": True}


def _tool_noop(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    return {"noop": True, "reason": str(args.get("reason", ""))[:500]}


TOOLS.register(ToolSpec("write_file", "Atomically write or append text content to a sandboxed file.", ToolRisk.GUARDED,
                        {"path": "string", "content": "string", "mode": "overwrite|append"}, _tool_write_file))
TOOLS.register(ToolSpec("read_file", "Read a sandboxed file, optionally by line range.", ToolRisk.SAFE,
                        {"path": "string", "start_line": "int", "end_line": "int"}, _tool_read_file))
TOOLS.register(ToolSpec("append_file", "Append lines with optional exact-line deduplication.", ToolRisk.GUARDED,
                        {"path": "string", "lines": "string[]", "unique": "bool"}, _tool_append_file))
TOOLS.register(ToolSpec("replace_lines", "Replace an inclusive line range with new content.", ToolRisk.GUARDED,
                        {"path": "string", "start_line": "int", "end_line": "int", "replacement": "string|string[]"},
                        _tool_replace_lines))
TOOLS.register(ToolSpec("check_lines", "Batched exact-line membership test against a file.", ToolRisk.SAFE,
                        {"path": "string", "lines": "string[]"}, _tool_check_lines))
TOOLS.register(ToolSpec("list_dir", "List sandbox directory entries up to a depth.", ToolRisk.SAFE,
                        {"path": "string", "depth": "int"}, _tool_list_dir))
TOOLS.register(ToolSpec("delete_path", "Delete a sandboxed file or directory tree.", ToolRisk.PRIVILEGED,
                        {"path": "string"}, _tool_delete_path))
TOOLS.register(ToolSpec("search_files", "Regex search across sandbox files.", ToolRisk.SAFE,
                        {"pattern": "string", "glob": "string"}, _tool_search_files))
TOOLS.register(ToolSpec("run_python", "Execute an isolated Python script inside the sandbox.", ToolRisk.PRIVILEGED,
                        {"code": "string", "timeout": "int"}, _tool_run_python))
TOOLS.register(ToolSpec("run_shell", "Execute a guarded shell command inside the sandbox.", ToolRisk.PRIVILEGED,
                        {"command": "string", "timeout": "int"}, _tool_run_shell))
TOOLS.register(ToolSpec("memory_search", "Hybrid RRF retrieval across skills and the knowledge wiki.", ToolRisk.SAFE,
                        {"query": "string", "limit": "int"}, _tool_memory_search))
TOOLS.register(ToolSpec("memory_write", "Create or revise a persistent knowledge wiki page.", ToolRisk.GUARDED,
                        {"slug": "string", "title": "string", "body": "string"}, _tool_memory_write))
TOOLS.register(ToolSpec("record_skill", "Persist a reusable procedural skill into experiential memory.", ToolRisk.GUARDED,
                        {"name": "string", "description": "string", "when_to_use": "string",
                         "procedure": "string[]", "tools": "string[]"}, _tool_record_skill))
TOOLS.register(ToolSpec("http_fetch", "Fetch an external HTTP(S) resource with SSRF guardrails.", ToolRisk.GUARDED,
                        {"url": "string", "method": "GET|POST|HEAD", "headers": "object", "body": "any"},
                        _tool_http_fetch))
TOOLS.register(ToolSpec("reason", "Invoke a bounded analytical sub-call over the current state.", ToolRisk.SAFE,
                        {"question": "string"}, _tool_reason))
TOOLS.register(ToolSpec("finish", "Terminate the run and emit the final answer.", ToolRisk.SAFE,
                        {"final_answer": "string"}, _tool_finish))
TOOLS.register(ToolSpec("noop", "Explicit no-operation step used to stabilize state.", ToolRisk.SAFE,
                        {"reason": "string"}, _tool_noop))


class WorkingMemory:
    def __init__(self, db: Database) -> None:
        self.db = db

    def derive(self, state: ExecutionState, spec: ProcedureSpec) -> Dict[str, Any]:
        open_goals = [g for g in state.subgoals if isinstance(g, dict) and not g.get("done")]
        done_goals = [g for g in state.subgoals if isinstance(g, dict) and g.get("done")]
        return {
            "phase": state.phase,
            "progress": round(state.progress, 3),
            "open_subgoals": [self._goal_view(g) for g in open_goals[:12]],
            "completed_subgoals": [self._goal_view(g) for g in done_goals[-6:]],
            "unresolved_blockers": state.blockers[:8],
            "environment_constraints": (state.constraints_observed + spec.constraints)[:10],
            "verified_facts": _truncate_mapping(state.facts, 2400),
            "artifacts": sorted(list(state.artifacts.keys()))[:24],
            "next_intent": state.next_intent[:600],
            "skill_hints": state.skill_hints[:6],
        }

    @staticmethod
    def _goal_view(goal: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(goal.get("id", ""))[:64],
            "goal": str(goal.get("goal", goal.get("title", "")))[:300],
            "done": bool(goal.get("done", False)),
            "depends_on": [str(d)[:64] for d in (goal.get("depends_on") or [])][:6],
        }

    def retrieval_signature(self, state: ExecutionState, spec: ProcedureSpec) -> str:
        wm = self.derive(state, spec)
        parts = [
            spec.title,
            spec.objective[:400],
            wm["phase"],
            wm["next_intent"],
            " ".join(g["goal"] for g in wm["open_subgoals"]),
            " ".join(wm["unresolved_blockers"]),
            " ".join(wm["skill_hints"]),
        ]
        return " ".join(p for p in parts if p)


class ExperientialMemory:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = threading.RLock()

    def upsert(
        self,
        tenant_id: str,
        name: str,
        description: str,
        trigger_signature: str,
        procedure: List[str],
        tools: List[str],
        reason: str,
        bump_version: bool = True,
    ) -> Skill:
        with self._lock:
            existing = self.db.query_one("SELECT * FROM skills WHERE tenant_id=? AND name=?", (tenant_id, name))
            text_blob = " ".join([name, description, trigger_signature] + procedure + tools)
            embedding = encode_vector(hashed_embedding(text_blob))
            now = iso()
            if existing:
                skill = Skill.from_row(existing)
                version = skill.version + 1 if bump_version else skill.version
                self.db.execute(
                    "UPDATE skills SET description=?, trigger_signature=?, procedure=?, tools=?, version=?,"
                    " embedding=?, updated_at=? WHERE skill_id=?",
                    (description, trigger_signature, jdump(procedure), jdump(tools), version, embedding, now,
                     skill.skill_id),
                )
                self._reindex(skill.skill_id, tenant_id, text_blob)
                self._snapshot(skill.skill_id, tenant_id, version, reason, True)
                return self.get(skill.skill_id) or skill
            skill_id = new_id("skl")
            self.db.execute(
                "INSERT INTO skills(skill_id, tenant_id, name, description, trigger_signature, procedure, tools,"
                " version, success_count, failure_count, quarantined, embedding, updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (skill_id, tenant_id, name, description, trigger_signature, jdump(procedure), jdump(tools),
                 1, 0, 0, 0, embedding, now),
            )
            self._reindex(skill_id, tenant_id, text_blob)
            self._snapshot(skill_id, tenant_id, 1, reason, True)
            created = self.get(skill_id)
            if created is None:
                raise RuntimeError("skill insert failed")
            self._write_disk(created)
            return created

    def _reindex(self, skill_id: str, tenant_id: str, text: str) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM skills_fts WHERE skill_id=?", (skill_id,))
            conn.execute("INSERT INTO skills_fts(skill_id, tenant_id, text) VALUES(?,?,?)",
                         (skill_id, tenant_id, text))

    def _snapshot(self, skill_id: str, tenant_id: str, version: int, reason: str, accepted: bool) -> None:
        row = self.db.query_one("SELECT * FROM skills WHERE skill_id=?", (skill_id,))
        snapshot = {k: row[k] for k in row.keys()} if row else {}
        self.db.execute(
            "INSERT INTO skill_versions(version_id, skill_id, tenant_id, version, snapshot, reason, accepted, created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (new_id("skv"), skill_id, tenant_id, version, jdump(snapshot), reason[:1000], 1 if accepted else 0, iso()),
        )

    def _write_disk(self, skill: Skill) -> None:
        folder = SKILL_ROOT / skill.tenant_id
        folder.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", skill.name)[:80] or skill.skill_id
        path = folder / f"{safe}.json"
        payload = {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "description": skill.description,
            "when_to_use": skill.trigger_signature,
            "procedure": skill.procedure,
            "tools": skill.tools,
            "version": skill.version,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(jdump(payload), encoding="utf-8")
        os.replace(tmp, path)

    def get(self, skill_id: str) -> Optional[Skill]:
        row = self.db.query_one("SELECT * FROM skills WHERE skill_id=?", (skill_id,))
        return Skill.from_row(row) if row else None

    def by_name(self, tenant_id: str, name: str) -> Optional[Skill]:
        row = self.db.query_one("SELECT * FROM skills WHERE tenant_id=? AND name=?", (tenant_id, name))
        return Skill.from_row(row) if row else None

    def list_skills(self, tenant_id: str, include_quarantined: bool = False) -> List[Skill]:
        if include_quarantined:
            rows = self.db.query("SELECT * FROM skills WHERE tenant_id=? ORDER BY name", (tenant_id,))
        else:
            rows = self.db.query("SELECT * FROM skills WHERE tenant_id=? AND quarantined=0 ORDER BY name", (tenant_id,))
        return [Skill.from_row(r) for r in rows]

    def retrieve(self, tenant_id: str, query: str, limit: int = 2) -> List[Skill]:
        candidates = self.list_skills(tenant_id)
        if not candidates:
            return []
        qvec = hashed_embedding(query)
        dense = sorted(candidates, key=lambda s: cosine(qvec, s.embedding), reverse=True)
        dense_ids = [s.skill_id for s in dense[: max(10, limit * 5)]]
        sparse_ids: List[str] = []
        fts_query = self._fts_query(query)
        if fts_query:
            try:
                rows = self.db.query(
                    "SELECT skill_id FROM skills_fts WHERE tenant_id=? AND skills_fts MATCH ?"
                    " ORDER BY bm25(skills_fts) LIMIT ?",
                    (tenant_id, fts_query, max(10, limit * 5)),
                )
                sparse_ids = [r["skill_id"] for r in rows]
            except sqlite3.OperationalError:
                sparse_ids = []
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
        by_id = {s.skill_id: s for s in candidates}
        out: List[Skill] = []
        for skill_id, _score in fused:
            skill = by_id.get(skill_id)
            if skill is None:
                continue
            out.append(skill)
            if len(out) >= limit:
                break
        if not out:
            out = dense[:limit]
        out.sort(key=lambda s: s.reliability, reverse=True)
        return out

    @staticmethod
    def _fts_query(query: str) -> str:
        toks = tokenize(query)[:16]
        if not toks:
            return ""
        return " OR ".join(f'"{t}"' for t in toks)

    def record_outcome(self, skill_id: str, success: bool) -> None:
        column = "success_count" if success else "failure_count"
        self.db.execute(f"UPDATE skills SET {column} = {column} + 1, updated_at=? WHERE skill_id=?", (iso(), skill_id))

    def set_quarantine(self, skill_id: str, quarantined: bool) -> None:
        self.db.execute("UPDATE skills SET quarantined=?, updated_at=? WHERE skill_id=?",
                        (1 if quarantined else 0, iso(), skill_id))

    def rollback(self, skill_id: str) -> bool:
        rows = self.db.query(
            "SELECT * FROM skill_versions WHERE skill_id=? AND accepted=1 ORDER BY version DESC LIMIT 2",
            (skill_id,),
        )
        if len(rows) < 2:
            return False
        snapshot = jload(rows[1]["snapshot"], {}) or {}
        if not snapshot:
            return False
        self.db.execute(
            "UPDATE skills SET description=?, trigger_signature=?, procedure=?, tools=?, version=?, embedding=?,"
            " updated_at=? WHERE skill_id=?",
            (
                snapshot.get("description", ""),
                snapshot.get("trigger_signature", ""),
                snapshot.get("procedure", "[]"),
                snapshot.get("tools", "[]"),
                int(snapshot.get("version", 1)),
                snapshot.get("embedding", ""),
                iso(),
                skill_id,
            ),
        )
        return True


class KnowledgeWiki:
    def __init__(self, db: Database, root: Path) -> None:
        self.db = db
        self.root = root
        self._lock = threading.RLock()
        self._git_init()

    def _tenant_dir(self, tenant_id: str) -> Path:
        path = self.root / tenant_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _git_init(self) -> None:
        if not GIT_ENABLED:
            return
        if (self.root / ".git").exists():
            return
        with contextlib.suppress(Exception):
            subprocess.run(["git", "init", "-q"], cwd=str(self.root), check=False, capture_output=True, timeout=30)
            subprocess.run(["git", "config", "user.email", "agent@runtime.local"], cwd=str(self.root),
                           check=False, capture_output=True, timeout=30)
            subprocess.run(["git", "config", "user.name", "Agent Runtime"], cwd=str(self.root),
                           check=False, capture_output=True, timeout=30)

    def _git_commit(self, message: str) -> Optional[str]:
        if not GIT_ENABLED:
            return None
        try:
            subprocess.run(["git", "add", "-A"], cwd=str(self.root), check=False, capture_output=True, timeout=60)
            proc = subprocess.run(
                ["git", "commit", "-q", "-m", message[:2000], "--allow-empty"],
                cwd=str(self.root), check=False, capture_output=True, timeout=60,
            )
            if proc.returncode not in (0, 1):
                return None
            rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(self.root), check=False,
                                 capture_output=True, text=True, timeout=30)
            return rev.stdout.strip() or None
        except Exception:
            return None

    def diff(self, limit: int = 1) -> str:
        if not GIT_ENABLED:
            return ""
        try:
            proc = subprocess.run(
                ["git", "diff", f"HEAD~{max(1, limit)}", "HEAD", "--unified=2"],
                cwd=str(self.root), check=False, capture_output=True, text=True, timeout=60,
            )
            return proc.stdout[:120_000]
        except Exception:
            return ""

    @staticmethod
    def slugify(text: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
        return base[:80] or "page"

    def upsert(self, tenant_id: str, slug: str, title: str, body: str, reason: str = "") -> Dict[str, Any]:
        slug = self.slugify(slug)
        with self._lock:
            existing = self.db.query_one("SELECT * FROM wiki_pages WHERE tenant_id=? AND slug=?", (tenant_id, slug))
            embedding = encode_vector(hashed_embedding(f"{title} {body}"))
            now = iso()
            if existing:
                revision = int(existing["revision"]) + 1
                self.db.execute(
                    "UPDATE wiki_pages SET title=?, body=?, revision=?, embedding=?, updated_at=? WHERE page_id=?",
                    (title, body, revision, embedding, now, existing["page_id"]),
                )
                page_id = existing["page_id"]
            else:
                page_id = new_id("wik")
                revision = 1
                self.db.execute(
                    "INSERT INTO wiki_pages(page_id, tenant_id, slug, title, body, revision, embedding, updated_at)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    (page_id, tenant_id, slug, title, body, revision, embedding, now),
                )
            with self.db.tx() as conn:
                conn.execute("DELETE FROM wiki_fts WHERE page_id=?", (page_id,))
                conn.execute("INSERT INTO wiki_fts(page_id, tenant_id, text) VALUES(?,?,?)",
                             (page_id, tenant_id, f"{title}\n{body}"))
            path = self._tenant_dir(tenant_id) / f"{slug}.md"
            content = f"# {title}\n\nrevision: {revision}\nupdated: {now}\n\n{body}\n"
            tmp = path.with_suffix(".md.tmp")
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)
            commit = self._git_commit(f"wiki({tenant_id}): {slug} r{revision} {reason}".strip())
            return {"page_id": page_id, "slug": slug, "title": title, "revision": revision, "commit": commit}

    def get(self, tenant_id: str, slug: str) -> Optional[Dict[str, Any]]:
        row = self.db.query_one("SELECT * FROM wiki_pages WHERE tenant_id=? AND slug=?", (tenant_id, self.slugify(slug)))
        return {k: row[k] for k in row.keys()} if row else None

    def list_pages(self, tenant_id: str) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT page_id, slug, title, revision, updated_at FROM wiki_pages WHERE tenant_id=? ORDER BY updated_at DESC",
            (tenant_id,),
        )
        return [{k: r[k] for k in r.keys()} for r in rows]

    def search(self, tenant_id: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        rows = self.db.query("SELECT * FROM wiki_pages WHERE tenant_id=?", (tenant_id,))
        pages = [{k: r[k] for k in r.keys()} for r in rows]
        if not pages:
            return []
        qvec = hashed_embedding(query)
        dense = sorted(pages, key=lambda p: cosine(qvec, decode_vector(p["embedding"])), reverse=True)
        dense_ids = [p["page_id"] for p in dense[: max(10, limit * 5)]]
        sparse_ids: List[str] = []
        toks = tokenize(query)[:16]
        if toks:
            match = " OR ".join(f'"{t}"' for t in toks)
            try:
                frows = self.db.query(
                    "SELECT page_id FROM wiki_fts WHERE tenant_id=? AND wiki_fts MATCH ? ORDER BY bm25(wiki_fts) LIMIT ?",
                    (tenant_id, match, max(10, limit * 5)),
                )
                sparse_ids = [r["page_id"] for r in frows]
            except sqlite3.OperationalError:
                sparse_ids = []
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
        by_id = {p["page_id"]: p for p in pages}
        out: List[Dict[str, Any]] = []
        for page_id, _score in fused:
            page = by_id.get(page_id)
            if page:
                out.append(page)
            if len(out) >= limit:
                break
        return out or dense[:limit]


class MemorySubsystem:
    def __init__(self, db: Database) -> None:
        self.wm = WorkingMemory(db)
        self.em = ExperientialMemory(db)
        self.wiki = KnowledgeWiki(db, WIKI_ROOT)
        self.db = db

    def route_skills(self, tenant_id: str, state: ExecutionState, spec: ProcedureSpec, limit: int = 2) -> List[Skill]:
        signature = self.wm.retrieval_signature(state, spec)
        return self.em.retrieve(tenant_id, signature, limit=limit)

    def route_caveats(self, tenant_id: str, state: ExecutionState, spec: ProcedureSpec, limit: int = 2) -> List[Dict[str, Any]]:
        signature = self.wm.retrieval_signature(state, spec)
        return self.wiki.search(tenant_id, signature, limit=limit)

    def record_trace(self, trace: ExecutionTrace) -> None:
        self.db.execute(
            "INSERT INTO traces(trace_id, run_id, step, pre_state_digest, selected_skill, tool, outcome,"
            " delta_digest, post_state_digest, receipt, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                trace.trace_id, trace.run_id, trace.step, trace.pre_state_digest, trace.selected_skill,
                trace.tool, trace.outcome, trace.delta_digest, trace.post_state_digest,
                jdump(trace.receipt), trace.created_at,
            ),
        )

    def run_traces(self, run_id: str, limit: int = 400) -> List[Dict[str, Any]]:
        rows = self.db.query("SELECT * FROM traces WHERE run_id=? ORDER BY step ASC LIMIT ?", (run_id, limit))
        return [{k: (jload(r[k], {}) if k == "receipt" else r[k]) for k in r.keys()} for r in rows]


MEMORY = MemorySubsystem(DB)


class PolicyPriorStore:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = threading.RLock()

    @staticmethod
    def signature(spec: ProcedureSpec, state: ExecutionState) -> str:
        raw = "|".join([spec.title[:60], state.phase[:40], (state.next_intent or "")[:80]])
        return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()

    def get(self, tenant_id: str, signature: str) -> Optional[Dict[str, Any]]:
        row = self.db.query_one("SELECT * FROM policy_priors WHERE tenant_id=? AND signature=?", (tenant_id, signature))
        return {k: row[k] for k in row.keys()} if row else None

    def update(self, tenant_id: str, signature: str, directive: str, gradient: float) -> None:
        with self._lock:
            existing = self.get(tenant_id, signature)
            if existing:
                logit = clamp(float(existing["logit"]) + gradient, -6.0, 6.0)
                self.db.execute(
                    "UPDATE policy_priors SET directive=?, logit=?, updates=updates+1, updated_at=? WHERE prior_id=?",
                    (directive[:2000], logit, iso(), existing["prior_id"]),
                )
            else:
                self.db.execute(
                    "INSERT INTO policy_priors(prior_id, tenant_id, signature, directive, logit, updates, updated_at)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (new_id("pri"), tenant_id, signature, directive[:2000], clamp(gradient, -6.0, 6.0), 1, iso()),
                )

    def active_directives(self, tenant_id: str, signature: str, limit: int = 3) -> List[str]:
        rows = self.db.query(
            "SELECT directive, logit FROM policy_priors WHERE tenant_id=? AND signature=? AND logit > 0"
            " ORDER BY logit DESC LIMIT ?",
            (tenant_id, signature, limit),
        )
        return [r["directive"] for r in rows if r["directive"]]

    def global_directives(self, tenant_id: str, limit: int = 3) -> List[str]:
        rows = self.db.query(
            "SELECT directive FROM policy_priors WHERE tenant_id=? AND logit > 0.8 ORDER BY logit DESC LIMIT ?",
            (tenant_id, limit),
        )
        return [r["directive"] for r in rows if r["directive"]]


PRIORS = PolicyPriorStore(DB)


class TokenLevelDistiller
    def __init__(self, db: Database, model: ModelClient) -> None:
        self.db = db
        self.model = model

    @staticmethod
    def _token_split(text: str) -> List[str]:
        return re.findall(r"\s*\S+", text)[:2048]

    @staticmethod
    def _distribution(tokens: Sequence[str], bias: Dict[str, float], temperature: float) -> List[float]:
        counts: Dict[str, float] = defaultdict(float)
        for t in tokens:
            counts[t.strip().lower()] += 1.0
        total = sum(counts.values()) or 1.0
        logprobs: List[float] = []
        for tok in tokens:
            key = tok.strip().lower()
            p = counts[key] / total
            adj = bias.get(key, 0.0)
            score = math.log(max(p, 1e-9)) / max(0.05, temperature) + adj
            logprobs.append(score)
        if not logprobs:
            return []
        m = max(logprobs)
        exps = [math.exp(s - m) for s in logprobs]
        z = sum(exps) or 1.0
        return [math.log(max(e / z, 1e-12)) for e in exps]

    @staticmethod
    def _bias_from_patch(patch: ReflectionPatch) -> Dict[str, float]:
        bias: Dict[str, float] = {}
        emphasis = " ".join([patch.root_cause, patch.guidance] + patch.pivot_actions)
        for tok in tokenize(emphasis):
            bias[tok] = bias.get(tok, 0.0) + 0.55
        for tok in tokenize(patch.failure_point):
            bias[tok] = bias.get(tok, 0.0) - 0.35
        return bias

    def distill(
        self,
        tenant_id: str,
        run_id: str,
        patch: ReflectionPatch,
        step_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not step_records:
            return {"samples": 0, "mean_reverse_kl": 0.0, "priors_updated": 0}
        bias = self._bias_from_patch(patch)
        samples = 0
        kl_sum = 0.0
        priors_updated = 0
        for record in step_records[-64:]:
            student_text = str(record.get("student_completion", ""))
            if not student_text.strip():
                continue
            tokens = self._token_split(student_text)
            if not tokens:
                continue
            student_lp = self._distribution(tokens, {}, temperature=1.0)
            teacher_lp = self._distribution(tokens, bias, temperature=0.75)
            if not student_lp or not teacher_lp:
                continue
            n = min(len(student_lp), len(teacher_lp))
            rkl = 0.0
            for i in range(n):
                q = math.exp(student_lp[i])
                rkl += q * (student_lp[i] - teacher_lp[i])
            rkl = abs(rkl)
            weight = clamp(1.0 if not patch.verdict else 0.35, 0.05, 1.0)
            self.db.execute(
                "INSERT INTO distillation_samples(sample_id, run_id, tenant_id, step, student_prompt_digest,"
                " teacher_prompt_digest, tokens, student_logprobs, teacher_logprobs, reverse_kl, weight, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    new_id("dst"), run_id, tenant_id, int(record.get("step", 0)),
                    stable_hash(record.get("student_prompt", "")),
                    stable_hash(patch.to_dict()),
                    jdump(tokens[:256]),
                    jdump([round(v, 6) for v in student_lp[:256]]),
                    jdump([round(v, 6) for v in teacher_lp[:256]]),
                    round(rkl, 8), round(weight, 4), iso(),
                ),
            )
            samples += 1
            kl_sum += rkl
            signature = str(record.get("signature", ""))
            if signature:
                directive = patch.guidance or (patch.pivot_actions[0] if patch.pivot_actions else "")
                if directive:
                    gradient = (0.45 if not patch.verdict else 0.12) * clamp(rkl * 4.0, 0.05, 1.0)
                    PRIORS.update(tenant_id, signature, directive, gradient)
                    priors_updated += 1
        mean_kl = kl_sum / samples if samples else 0.0
        return {"samples": samples, "mean_reverse_kl": round(mean_kl, 8), "priors_updated": priors_updated}


DISTILLER = TokenLevelDistiller(DB, MODEL)


class Verifier:
    SAFE_BUILTINS = {
        "len": len, "str": str, "int": int, "float": float, "bool": bool, "abs": abs,
        "min": min, "max": max, "sum": sum, "any": any, "all": all, "sorted": sorted,
        "round": round, "list": list, "dict": dict, "set": set, "tuple": tuple,
        "enumerate": enumerate, "range": range, "isinstance": isinstance, "zip": zip,
    }

    @classmethod
    def verify(cls, spec: ProcedureSpec, state: ExecutionState, sandbox: FileSystemSandbox,
               final_answer: str) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []
        for criterion in spec.success_criteria:
            checks.append(cls._check_criterion(criterion, state, sandbox, final_answer))
        program_result: Optional[Dict[str, Any]] = None
        if spec.verifier_program and spec.verifier_program.strip():
            program_result = cls._run_program(spec.verifier_program, state, sandbox, final_answer)
            checks.append({"criterion": "verifier_program", "passed": bool(program_result.get("passed")),
                           "detail": program_result.get("detail", "")})
        if not checks:
            passed = bool(final_answer and final_answer.strip())
            checks.append({"criterion": "non_empty_final_answer", "passed": passed,
                           "detail": "final answer present" if passed else "final answer empty"})
        overall = all(c["passed"] for c in checks)
        return {"passed": overall, "checks": checks, "program": program_result}

    @classmethod
    def _check_criterion(cls, criterion: str, state: ExecutionState, sandbox: FileSystemSandbox,
                         final_answer: str) -> Dict[str, Any]:
        text = criterion.strip()
        lowered = text.lower()
        if lowered.startswith("file_exists:"):
            rel = text.split(":", 1)[1].strip()
            try:
                path = sandbox.resolve(rel)
                ok = path.exists()
                return {"criterion": text, "passed": ok, "detail": f"exists={ok}"}
            except ValueError as exc:
                return {"criterion": text, "passed": False, "detail": str(exc)}
        if lowered.startswith("file_contains:"):
            body = text.split(":", 1)[1]
            if "::" in body:
                rel, needle = body.split("::", 1)
            else:
                parts = body.split(None, 1)
                rel, needle = (parts + [""])[:2]
            try:
                path = sandbox.resolve(rel.strip())
                content = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
                ok = needle.strip() in content
                return {"criterion": text, "passed": ok, "detail": f"needle_found={ok}"}
            except (ValueError, OSError) as exc:
                return {"criterion": text, "passed": False, "detail": str(exc)}
        if lowered.startswith("state_fact:"):
            key = text.split(":", 1)[1].strip()
            ok = key in state.facts and state.facts.get(key) not in (None, "", [], {})
            return {"criterion": text, "passed": ok, "detail": f"fact_present={ok}"}
        if lowered.startswith("artifact:"):
            key = text.split(":", 1)[1].strip()
            ok = key in state.artifacts
            return {"criterion": text, "passed": ok, "detail": f"artifact_present={ok}"}
        if lowered.startswith("answer_contains:"):
            needle = text.split(":", 1)[1].strip().lower()
            ok = needle in (final_answer or "").lower()
            return {"criterion": text, "passed": ok, "detail": f"answer_match={ok}"}
        if lowered.startswith("progress_at_least:"):
            try:
                threshold = float(text.split(":", 1)[1].strip())
            except ValueError:
                threshold = 1.0
            ok = state.progress >= threshold
            return {"criterion": text, "passed": ok, "detail": f"progress={state.progress}"}
        if lowered.startswith("no_blockers"):
            ok = not state.blockers
            return {"criterion": text, "passed": ok, "detail": f"blockers={len(state.blockers)}"}
        if lowered.startswith("all_subgoals_done"):
            ok = bool(state.subgoals) and all(bool(g.get("done")) for g in state.subgoals if isinstance(g, dict))
            return {"criterion": text, "passed": ok, "detail": f"subgoals={len(state.subgoals)}"}
        keywords = [k for k in tokenize(text) if len(k) > 3][:8]
        haystack = (final_answer or "") + " " + jdump(state.to_dict())
        hay_tokens = set(tokenize(haystack))
        matched = sum(1 for k in keywords if k in hay_tokens)
        ok = bool(keywords) and matched >= max(1, int(len(keywords) * 0.6))
        return {"criterion": text, "passed": ok, "detail": f"keyword_coverage={matched}/{len(keywords)}"}

    @classmethod
    def _run_program(cls, program: str, state: ExecutionState, sandbox: FileSystemSandbox,
                     final_answer: str) -> Dict[str, Any]:
        env: Dict[str, Any] = {
            "__builtins__": dict(cls.SAFE_BUILTINS),
            "state": json.loads(jdump(state.to_dict())),
            "final_answer": final_answer,
            "read_text": lambda p: (sandbox.resolve(p).read_text(encoding="utf-8", errors="replace")
                                    if sandbox.resolve(p).exists() else ""),
            "path_exists": lambda p: sandbox.resolve(p).exists(),
            "list_files": lambda p=".": [e["path"] for e in sandbox.list_dir(p, 3)["entries"]],
            "result": {},
        }
        try:
            compiled = compile(program, "<verifier>", "exec")
            exec(compiled, env, env)
            result = env.get("result")
            if isinstance(result, dict):
                return {"passed": bool(result.get("passed")), "detail": str(result.get("detail", ""))[:4000]}
            return {"passed": bool(result), "detail": "verifier returned non-dict result"}
        except Exception as exc:
            return {"passed": False, "detail": f"verifier error: {type(exc).__name__}: {exc}"}


class ReflectionEngine:
    def __init__(self, db: Database, model: ModelClient, memory: MemorySubsystem) -> None:
        self.db = db
        self.model = model
        self.memory = memory

    def build_patch(self, tenant: Tenant, spec: ProcedureSpec, run_id: str, verdict: Dict[str, Any],
                    state: ExecutionState, traces: List[Dict[str, Any]]) -> ReflectionPatch:
        failed_checks = [c for c in verdict.get("checks", []) if not c.get("passed")]
        failing_traces = [t for t in traces if t.get("outcome") != "ok"][-8:]
        heuristic = self._heuristic_patch(run_id, verdict, failed_checks, failing_traces, state)
        if not self.model.available:
            self._persist(tenant.tenant_id, heuristic)
            return heuristic
        prompt = {
            "task": spec.frozen_view(),
            "verification": {"passed": verdict.get("passed"), "failed_checks": failed_checks[:8]},
            "terminal_state": state.compact(3000),
            "recent_failures": [
                {"step": t.get("step"), "tool": t.get("tool"), "outcome": t.get("outcome"),
                 "receipt": str(t.get("receipt"))[:600]}
                for t in failing_traces
            ],
            "required_json_schema": {
                "failure_point": "string",
                "root_cause": "string",
                "pivot_actions": ["string"],
                "memory_target": "working_memory|experiential_memory|knowledge_wiki|tooling|specification",
                "guidance": "string",
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a deterministic post-hoc diagnostician for an autonomous agent runtime. "
                    "Emit ONLY one JSON object matching the required schema. No markdown, no prose outside JSON. "
                    "Be specific and actionable. Never invent facts absent from the evidence."
                ),
            },
            {"role": "user", "content": jdump(prompt)},
        ]
        try:
            text, tokens = self.model.complete(messages, max_tokens=min(6000, MODEL_MAX_TOKENS), temperature=0.3)
            with contextlib.suppress(BudgetExceeded, AuthorizationDenied):
                TENANTS.charge(tenant.tenant_id, tokens)
            parsed = GrammarDecoder.extract_json(text)
            pivots = parsed.get("pivot_actions", [])
            if isinstance(pivots, str):
                pivots = [pivots]
            patch = ReflectionPatch(
                patch_id=new_id("ref"),
                run_id=run_id,
                verdict=bool(verdict.get("passed")),
                failure_point=str(parsed.get("failure_point", heuristic.failure_point))[:2000],
                root_cause=str(parsed.get("root_cause", heuristic.root_cause))[:4000],
                pivot_actions=[str(p)[:600] for p in pivots][:8] or heuristic.pivot_actions,
                memory_target=self._normalize_target(str(parsed.get("memory_target", heuristic.memory_target))),
                guidance=str(parsed.get("guidance", heuristic.guidance))[:4000],
            )
        except (SchemaError, Exception) as exc:
            LOG.warning("reflection model call failed, using heuristic: %s", exc)
            patch = heuristic
        self._persist(tenant.tenant_id, patch)
        return patch

    @staticmethod
    def _normalize_target(value: str) -> str:
        allowed = {"working_memory", "experiential_memory", "knowledge_wiki", "tooling", "specification"}
        v = value.strip().lower().replace(" ", "_")
        return v if v in allowed else "experiential_memory"

    def _heuristic_patch(self, run_id: str, verdict: Dict[str, Any], failed_checks: List[Dict[str, Any]],
                         failing_traces: List[Dict[str, Any]], state: ExecutionState) -> ReflectionPatch:
        if verdict.get("passed"):
            failure_point = "none"
            root_cause = "trajectory satisfied all verifier checks"
            pivots = ["preserve current procedure ordering", "promote successful skill sequence"]
            guidance = "Reinforce the executed procedure; record it as a reusable skill."
            target = "experiential_memory"
        else:
            first = failed_checks[0]["criterion"] if failed_checks else "unspecified criterion"
            failure_point = f"step {failing_traces[-1]['step']} tool {failing_traces[-1]['tool']}" if failing_traces \
                else f"terminal verification: {first}"
            tool_errors = [str(t.get("receipt", {}).get("error", "")) for t in failing_traces if t.get("receipt")]
            root_cause = "; ".join([e for e in tool_errors if e][:3]) or f"unmet criterion: {first}"
            pivots = [
                f"re-attempt the failing objective with an explicit precondition check for: {first}",
                "decompose the blocking subgoal into verifiable atomic steps",
            ]
            if state.blockers:
                pivots.append(f"resolve persistent blocker: {state.blockers[0][:200]}")
            guidance = (
                f"Before terminating, deterministically assert '{first}'. "
                "If a tool errors twice consecutively, switch strategy instead of retrying identically."
            )
            target = "experiential_memory" if failing_traces else "specification"
        return ReflectionPatch(
            patch_id=new_id("ref"),
            run_id=run_id,
            verdict=bool(verdict.get("passed")),
            failure_point=failure_point[:2000],
            root_cause=root_cause[:4000],
            pivot_actions=pivots[:8],
            memory_target=target,
            guidance=guidance[:4000],
        )

    def _persist(self, tenant_id: str, patch: ReflectionPatch) -> None:
        self.db.execute(
            "INSERT INTO reflections(patch_id, run_id, tenant_id, verdict, failure_point, root_cause,"
            " pivot_actions, memory_target, guidance, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (patch.patch_id, patch.run_id, tenant_id, 1 if patch.verdict else 0, patch.failure_point,
             patch.root_cause, jdump(patch.pivot_actions), patch.memory_target, patch.guidance, patch.created_at),
        )

    def latest(self, run_id: str) -> Optional[ReflectionPatch]:
        row = self.db.query_one("SELECT * FROM reflections WHERE run_id=? ORDER BY created_at DESC LIMIT 1", (run_id,))
        if not row:
            return None
        return ReflectionPatch(
            patch_id=row["patch_id"], run_id=row["run_id"], verdict=bool(row["verdict"]),
            failure_point=row["failure_point"], root_cause=row["root_cause"],
            pivot_actions=jload(row["pivot_actions"], []) or [], memory_target=row["memory_target"],
            guidance=row["guidance"], created_at=row["created_at"],
        )


REFLECTOR = ReflectionEngine(DB, MODEL, MEMORY)


class RegressionGate:
    def __init__(self, db: Database, memory: MemorySubsystem) -> None:
        self.db = db
        self.memory = memory

    def diagnostics(self, tenant_id: str) -> List[Dict[str, Any]]:
        rows = self.db.query("SELECT * FROM diagnostic_tasks WHERE tenant_id=? ORDER BY created_at", (tenant_id,))
        return [{"task_id": r["task_id"], "name": r["name"], "payload": jload(r["payload"], {}),
                 "expectation": jload(r["expectation"], {})} for r in rows]

    def add_diagnostic(self, tenant_id: str, name: str, payload: Dict[str, Any], expectation: Dict[str, Any]) -> str:
        task_id = new_id("dgt")
        self.db.execute(
            "INSERT INTO diagnostic_tasks(task_id, tenant_id, name, payload, expectation, created_at)"
            " VALUES(?,?,?,?,?,?)",
            (task_id, tenant_id, name, jdump(payload), jdump(expectation), iso()),
        )
        return task_id

    def evaluate_skill(self, skill: Skill, diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        blob = " ".join([skill.name, skill.description, skill.trigger_signature] + skill.procedure + skill.tools).lower()
        vec = hashed_embedding(blob)
        for task in diagnostics:
            payload = task.get("payload", {})
            expectation = task.get("expectation", {})
            query = str(payload.get("query", payload.get("objective", task.get("name", ""))))
            relevance = cosine(vec, hashed_embedding(query))
            must = [str(m).lower() for m in expectation.get("must_include", [])]
            forbid = [str(m).lower() for m in expectation.get("must_not_include", [])]
            min_rel = float(expectation.get("min_relevance", 0.0))
            missing = [m for m in must if m not in blob]
            violated = [f for f in forbid if f in blob]
            passed = not missing and not violated and relevance >= min_rel
            results.append({
                "task": task.get("name"),
                "passed": passed,
                "relevance": round(relevance, 4),
                "missing": missing,
                "violated": violated,
            })
        structural = self._structural_checks(skill)
        results.extend(structural)
        score = sum(1 for r in results if r["passed"]) / len(results) if results else 1.0
        return {"score": round(score, 4), "passed": all(r["passed"] for r in results), "results": results}

    @staticmethod
    def _structural_checks(skill: Skill) -> List[Dict[str, Any]]:
        checks: List[Dict[str, Any]] = []
        checks.append({"task": "has_name", "passed": bool(skill.name.strip()), "relevance": 1.0,
                       "missing": [], "violated": []})
        checks.append({"task": "has_trigger", "passed": len(skill.trigger_signature.strip()) >= 8, "relevance": 1.0,
                       "missing": [], "violated": []})
        checks.append({"task": "procedure_nonempty", "passed": len([p for p in skill.procedure if str(p).strip()]) >= 1,
                       "relevance": 1.0, "missing": [], "violated": []})
        known = set(TOOLS.names())
        unknown = [t for t in skill.tools if t not in known]
        checks.append({"task": "tools_registered", "passed": not unknown, "relevance": 1.0,
                       "missing": unknown, "violated": []})
        checks.append({"task": "procedure_bounded", "passed": len(skill.procedure) <= 32, "relevance": 1.0,
                       "missing": [], "violated": []})
        return checks


GATE = RegressionGate(DB, MEMORY)


class MetaAgent:
    def __init__(self, db: Database, model: ModelClient, memory: MemorySubsystem, gate: RegressionGate) -> None:
        self.db = db
        self.model = model
        self.memory = memory
        self.gate = gate
        self._lock = threading.RLock()

    def consolidate(self, tenant: Tenant, spec: ProcedureSpec, run_id: str, patch: ReflectionPatch,
                    state: ExecutionState, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            report: Dict[str, Any] = {
                "run_id": run_id,
                "memory_target": patch.memory_target,
                "wiki": None,
                "skill": None,
                "gate": None,
                "rolled_back": False,
                "accepted": False,
            }
            wiki_result = self._update_wiki(tenant.tenant_id, spec, patch, state, traces)
            report["wiki"] = wiki_result
            proposal = self._propose_skill(tenant, spec, patch, state, traces)
            if proposal is None:
                report["accepted"] = True
                LEDGER.append(tenant.tenant_id, "consolidation", report, run_id)
                return report
            report["skill"] = {"name": proposal["name"], "action": proposal["action"]}
            diagnostics = self.gate.diagnostics(tenant.tenant_id)
            previous = self.memory.em.by_name(tenant.tenant_id, proposal["name"])
            baseline = self.gate.evaluate_skill(previous, diagnostics) if previous else {"score": 0.0, "passed": True}
            candidate = Skill(
                skill_id=previous.skill_id if previous else "candidate",
                tenant_id=tenant.tenant_id,
                name=proposal["name"],
                description=proposal["description"],
                trigger_signature=proposal["when_to_use"],
                procedure=proposal["procedure"],
                tools=proposal["tools"],
                version=(previous.version + 1) if previous else 1,
                success_count=previous.success_count if previous else 0,
                failure_count=previous.failure_count if previous else 0,
                embedding=hashed_embedding(" ".join([proposal["name"], proposal["description"],
                                                     proposal["when_to_use"]] + proposal["procedure"])),
            )
            evaluation = self.gate.evaluate_skill(candidate, diagnostics)
            report["gate"] = {"candidate": evaluation, "baseline_score": baseline.get("score", 0.0)}
            regression = evaluation["score"] + 1e-9 < float(baseline.get("score", 0.0))
            if not evaluation["passed"] or regression:
                report["accepted"] = False
                report["rolled_back"] = True
                if previous:
                    self.memory.em._snapshot(previous.skill_id, tenant.tenant_id, previous.version,
                                             f"rejected patch from {run_id}", False)
                LEDGER.append(tenant.tenant_id, "consolidation_rejected", report, run_id)
                return report
            stored = self.memory.em.upsert(
                tenant_id=tenant.tenant_id,
                name=candidate.name,
                description=candidate.description,
                trigger_signature=candidate.trigger_signature,
                procedure=candidate.procedure,
                tools=candidate.tools,
                reason=f"meta-agent consolidation from {run_id}: {patch.root_cause[:200]}",
            )
            self.memory.em._write_disk(stored)
            self.memory.em.record_outcome(stored.skill_id, bool(patch.verdict))
            report["accepted"] = True
            report["skill"]["skill_id"] = stored.skill_id
            report["skill"]["version"] = stored.version
            LEDGER.append(tenant.tenant_id, "consolidation_accepted", report, run_id)
            return report

    def _update_wiki(self, tenant_id: str, spec: ProcedureSpec, patch: ReflectionPatch, state: ExecutionState,
                     traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        slug = KnowledgeWiki.slugify(spec.title or "general-playbook")
        existing = self.memory.wiki.get(tenant_id, slug)
        header = f"Playbook: {spec.title}"
        entry_lines = [
            f"## Epoch {iso()}",
            f"- run: {patch.run_id}",
            f"- outcome: {'success' if patch.verdict else 'failure'}",
            f"- failure_point: {patch.failure_point}",
            f"- root_cause: {patch.root_cause}",
            f"- memory_target: {patch.memory_target}",
            "- pivot_actions:",
        ]
        entry_lines.extend(f"  - {p}" for p in patch.pivot_actions[:6])
        entry_lines.append(f"- guidance: {patch.guidance}")
        failed_tools = sorted({t.get("tool", "") for t in traces if t.get("outcome") != "ok"})
        if failed_tools:
            entry_lines.append(f"- unstable_tools: {', '.join(t for t in failed_tools if t)}")
        if state.constraints_observed:
            entry_lines.append(f"- environment_caveats: {'; '.join(state.constraints_observed[:5])}")
        entry = "\n".join(entry_lines)
        body = (existing["body"] + "\n\n" + entry) if existing else entry
        if len(body) > 400_000:
            body = body[-400_000:]
        result = self.memory.wiki.upsert(tenant_id, slug, header, body, reason=f"consolidate {patch.run_id}")
        result["diff"] = self.memory.wiki.diff(1)[:8000]
        return result

    def _propose_skill(self, tenant: Tenant, spec: ProcedureSpec, patch: ReflectionPatch, state: ExecutionState,
                       traces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        successful = [t for t in traces if t.get("outcome") == "ok"]
        if not successful and not patch.pivot_actions:
            return None
        heuristic_name = re.sub(r"[^a-z0-9 ]", "", (spec.title or "task").lower()).strip().replace(" ", "_")[:48]
        heuristic_name = heuristic_name or "general_procedure"
        used_tools: List[str] = []
        for t in successful:
            tool = t.get("tool")
            if tool and tool not in used_tools and tool in TOOLS.names():
                used_tools.append(tool)
        heuristic_procedure = [f"Use {t} to advance the objective with verified preconditions." for t in used_tools[:8]]
        heuristic_procedure.extend(patch.pivot_actions[:4])
        if not heuristic_procedure:
            heuristic_procedure = ["Decompose the objective into verifiable subgoals before acting."]
        fallback = {
            "action": "upsert",
            "name": heuristic_name,
            "description": f"Procedure distilled from run {patch.run_id} for objective: {spec.objective[:400]}",
            "when_to_use": f"{spec.title}: {state.phase}; {patch.failure_point[:200]}",
            "procedure": [str(p)[:400] for p in heuristic_procedure][:16],
            "tools": used_tools[:12] or ["reason"],
        }
        if not self.model.available:
            return fallback
        prompt = {
            "objective": spec.frozen_view(),
            "reflection": patch.to_dict(),
            "terminal_state": state.compact(2500),
            "successful_tool_sequence": [{"step": t.get("step"), "tool": t.get("tool")} for t in successful[:24]],
            "registered_tools": TOOLS.names(),
            "existing_skills": [s.name for s in self.memory.em.list_skills(tenant.tenant_id)][:60],
            "required_json_schema": {
                "action": "upsert",
                "name": "snake_case_identifier",
                "description": "string",
                "when_to_use": "string",
                "procedure": ["ordered imperative steps"],
                "tools": ["registered tool names only"],
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an isolated meta-agent that authors minimal, scoped procedural skill patches for an "
                    "agent skill library. You never modify yourself. Output exactly one JSON object matching the "
                    "schema, no markdown. Tools must be drawn only from registered_tools. Keep procedures under 16 steps."
                ),
            },
            {"role": "user", "content": jdump(prompt)},
        ]
        try:
            text, tokens = self.model.complete(messages, max_tokens=min(6000, MODEL_MAX_TOKENS), temperature=0.35)
            with contextlib.suppress(BudgetExceeded, AuthorizationDenied):
                TENANTS.charge(tenant.tenant_id, tokens)
            parsed = GrammarDecoder.extract_json(text)
            name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(parsed.get("name", heuristic_name))).strip("_").lower()[:48]
            procedure = parsed.get("procedure", [])
            if isinstance(procedure, str):
                procedure = [ln for ln in procedure.splitlines() if ln.strip()]
            tools = parsed.get("tools", [])
            if isinstance(tools, str):
                tools = [tools]
            registered = set(TOOLS.names())
            tools = [str(t) for t in tools if str(t) in registered]
            if not name or not procedure:
                return fallback
            return {
                "action": "upsert",
                "name": name,
                "description": str(parsed.get("description", fallback["description"]))[:2000],
                "when_to_use": str(parsed.get("when_to_use", fallback["when_to_use"]))[:1200],
                "procedure": [str(p)[:400] for p in procedure][:16],
                "tools": tools[:12] or fallback["tools"],
            }
        except Exception as exc:
            LOG.warning("meta-agent proposal failed, using heuristic: %s", exc)
            return fallback


META = MetaAgent(DB, MODEL, MEMORY, GATE)


class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def subscribe(self, topic: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=2048)
        async with self._lock:
            self._subs[topic].add(queue)
        return queue

    async def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subs[topic].discard(queue)
            if not self._subs[topic]:
                self._subs.pop(topic, None)

    def publish_threadsafe(self, topic: str, payload: Dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._deliver, topic, payload)
        except RuntimeError:
            pass

    def _deliver(self, topic: str, payload: Dict[str, Any]) -> None:
        for queue in list(self._subs.get(topic, set())) + list(self._subs.get("*", set())):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(payload)


BUS = EventBus()


class CheckpointStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, run_id: str, step: int, node: NodeKind, state: ExecutionState, observation: Observation,
             status: RunState, tokens_used: int) -> str:
        checkpoint_id = new_id("ckp")
        state_blob = jdump(state.to_dict())
        obs_blob = jdump(observation.to_dict())
        digest = stable_hash({"run": run_id, "step": step, "node": node.value, "state": state_blob, "obs": obs_blob})
        self.db.execute(
            "INSERT INTO checkpoints(checkpoint_id, run_id, step, node, state, observation, status, tokens_used,"
            " digest, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (checkpoint_id, run_id, step, node.value, state_blob, obs_blob, status.value, tokens_used, digest, iso()),
        )
        return checkpoint_id

    def latest(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.query_one(
            "SELECT * FROM checkpoints WHERE run_id=? ORDER BY step DESC, rowid DESC LIMIT 1", (run_id,)
        )
        if not row:
            return None
        return {
            "checkpoint_id": row["checkpoint_id"],
            "step": int(row["step"]),
            "node": row["node"],
            "state": jload(row["state"], {}) or {},
            "observation": jload(row["observation"], {}) or {},
            "status": row["status"],
            "tokens_used": int(row["tokens_used"]),
            "digest": row["digest"],
            "created_at": row["created_at"],
        }

    def history(self, run_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT checkpoint_id, step, node, status, tokens_used, digest, created_at FROM checkpoints"
            " WHERE run_id=? ORDER BY step ASC LIMIT ?",
            (run_id, limit),
        )
        return [{k: r[k] for k in r.keys()} for r in rows]

    def prune(self, run_id: str, keep: int = 400) -> int:
        rows = self.db.query("SELECT checkpoint_id FROM checkpoints WHERE run_id=? ORDER BY rowid DESC", (run_id,))
        stale = [r["checkpoint_id"] for r in rows[keep:]]
        if not stale:
            return 0
        with self.db.tx() as conn:
            conn.executemany("DELETE FROM checkpoints WHERE checkpoint_id=?", [(cid,) for cid in stale])
        return len(stale)


CHECKPOINTS = CheckpointStore(DB)


class RunRecord:
    __slots__ = ("run_id", "tenant_id", "spec_id", "conversation_id", "status", "node", "step", "tokens_used",
                 "retries", "state", "last_observation", "final_answer", "error", "created_at", "updated_at")

    def __init__(self, row: sqlite3.Row) -> None:
        self.run_id = row["run_id"]
        self.tenant_id = row["tenant_id"]
        self.spec_id = row["spec_id"]
        self.conversation_id = row["conversation_id"]
        self.status = RunState(row["status"])
        self.node = NodeKind(row["node"])
        self.step = int(row["step"])
        self.tokens_used = int(row["tokens_used"])
        self.retries = int(row["retries"])
        self.state = ExecutionState.from_dict(jload(row["state"], {}) or {})
        self.last_observation = jload(row["last_observation"], {}) or {}
        self.final_answer = row["final_answer"]
        self.error = row["error"]
        self.created_at = row["created_at"]
        self.updated_at = row["updated_at"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "spec_id": self.spec_id,
            "conversation_id": self.conversation_id,
            "status": self.status.value,
            "node": self.node.value,
            "step": self.step,
            "tokens_used": self.tokens_used,
            "retries": self.retries,
            "state": self.state.to_dict(),
            "last_observation": self.last_observation,
            "final_answer": self.final_answer,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class RunStore:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._lock = threading.RLock()

    def create(self, tenant_id: str, spec: ProcedureSpec, conversation_id: str = "") -> RunRecord:
        run_id = new_id("run")
        now = iso()
        state = ExecutionState(
            phase="bootstrap",
            next_intent="Analyze the objective and construct an initial verifiable subgoal decomposition.",
        )
        self.db.execute(
            "INSERT INTO runs(run_id, tenant_id, spec_id, conversation_id, status, node, step, tokens_used, retries,"
            " state, last_observation, final_answer, error, lease_owner, lease_expires_at, created_at, updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, tenant_id, spec.spec_id, conversation_id, RunState.PENDING.value, NodeKind.PERCEIVE.value,
             0, 0, 0, jdump(state.to_dict()), "{}", "", "", "", 0.0, now, now),
        )
        record = self.get(run_id)
        if record is None:
            raise RuntimeError("run creation failed")
        return record

    def get(self, run_id: str) -> Optional[RunRecord]:
        row = self.db.query_one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        return RunRecord(row) if row else None

    def get_scoped(self, tenant_id: str, run_id: str) -> Optional[RunRecord]:
        row = self.db.query_one("SELECT * FROM runs WHERE run_id=? AND tenant_id=?", (run_id, tenant_id))
        return RunRecord(row) if row else None

    def list_runs(self, tenant_id: str, status: Optional[str] = None, limit: int = 100) -> List[RunRecord]:
        if status:
            rows = self.db.query(
                "SELECT * FROM runs WHERE tenant_id=? AND status=? ORDER BY updated_at DESC LIMIT ?",
                (tenant_id, status, limit),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM runs WHERE tenant_id=? ORDER BY updated_at DESC LIMIT ?", (tenant_id, limit)
            )
        return [RunRecord(r) for r in rows]

    def resumable(self) -> List[RunRecord]:
        now = time.time()
        rows = self.db.query(
            "SELECT * FROM runs WHERE status IN (?,?,?) AND lease_expires_at < ? ORDER BY updated_at ASC",
            (RunState.RUNNING.value, RunState.PENDING.value, RunState.RECOVERING.value, now),
        )
        return [RunRecord(r) for r in rows]

    def acquire_lease(self, run_id: str, owner: str, ttl: float = 45.0) -> bool:
        with self._lock:
            now = time.time()
            with self.db.tx() as conn:
                cur = conn.execute(
                    "UPDATE runs SET lease_owner=?, lease_expires_at=? WHERE run_id=? AND"
                    " (lease_owner=? OR lease_expires_at < ?)",
                    (owner, now + ttl, run_id, owner, now),
                )
                return cur.rowcount > 0

    def renew_lease(self, run_id: str, owner: str, ttl: float = 45.0) -> None:
        self.db.execute(
            "UPDATE runs SET lease_expires_at=? WHERE run_id=? AND lease_owner=?",
            (time.time() + ttl, run_id, owner),
        )

    def release_lease(self, run_id: str, owner: str) -> None:
        self.db.execute(
            "UPDATE runs SET lease_owner='', lease_expires_at=0 WHERE run_id=? AND lease_owner=?", (run_id, owner)
        )

    def persist(self, run_id: str, *, status: Optional[RunState] = None, node: Optional[NodeKind] = None,
                step: Optional[int] = None, tokens_used: Optional[int] = None, retries: Optional[int] = None,
                state: Optional[ExecutionState] = None, observation: Optional[Observation] = None,
                final_answer: Optional[str] = None, error: Optional[str] = None) -> None:
        sets: List[str] = ["updated_at=?"]
        params: List[Any] = [iso()]
        if status is not None:
            sets.append("status=?")
            params.append(status.value)
        if node is not None:
            sets.append("node=?")
            params.append(node.value)
        if step is not None:
            sets.append("step=?")
            params.append(int(step))
        if tokens_used is not None:
            sets.append("tokens_used=?")
            params.append(int(tokens_used))
        if retries is not None:
            sets.append("retries=?")
            params.append(int(retries))
        if state is not None:
            sets.append("state=?")
            params.append(jdump(state.to_dict()))
        if observation is not None:
            sets.append("last_observation=?")
            params.append(jdump(observation.to_dict()))
        if final_answer is not None:
            sets.append("final_answer=?")
            params.append(final_answer[:200_000])
        if error is not None:
            sets.append("error=?")
            params.append(error[:8000])
        params.append(run_id)
        self.db.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id=?", params)

    def set_status(self, run_id: str, status: RunState) -> None:
        self.persist(run_id, status=status)

    def request_cancel(self, run_id: str) -> None:
        self.persist(run_id, status=RunState.CANCELLED)


RUNS = RunStore(DB)


class SpecStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, tenant_id: str, title: str, objective: str, constraints: List[str],
               success_criteria: List[str], allowed_tools: List[str], max_steps: int, token_budget: int,
               verifier_program: Optional[str]) -> ProcedureSpec:
        registered = set(TOOLS.names())
        tools = [t for t in allowed_tools if t in registered] or [t for t in registered]
        spec = ProcedureSpec(
            spec_id=new_id("spc"),
            tenant_id=tenant_id,
            title=title[:300] or "Untitled objective",
            objective=objective[:20000],
            constraints=[str(c)[:600] for c in constraints][:32],
            success_criteria=[str(c)[:600] for c in success_criteria][:32],
            allowed_tools=sorted(set(tools)),
            max_steps=int(clamp(float(max_steps or DEFAULT_MAX_STEPS), 1, 5000)),
            token_budget=int(max(1000, token_budget or DEFAULT_TOKEN_BUDGET)),
            verifier_program=(verifier_program or "")[:40000] or None,
        )
        row = spec.to_row()
        self.db.execute(
            "INSERT INTO specs(spec_id, tenant_id, title, objective, constraints, success_criteria, allowed_tools,"
            " max_steps, token_budget, verifier_program, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (row["spec_id"], row["tenant_id"], row["title"], row["objective"], row["constraints"],
             row["success_criteria"], row["allowed_tools"], row["max_steps"], row["token_budget"],
             row["verifier_program"], row["created_at"]),
        )
        return spec

    def get(self, spec_id: str) -> Optional[ProcedureSpec]:
        row = self.db.query_one("SELECT * FROM specs WHERE spec_id=?", (spec_id,))
        return ProcedureSpec.from_row(row) if row else None

    def get_scoped(self, tenant_id: str, spec_id: str) -> Optional[ProcedureSpec]:
        row = self.db.query_one("SELECT * FROM specs WHERE spec_id=? AND tenant_id=?", (spec_id, tenant_id))
        return ProcedureSpec.from_row(row) if row else None

    def list_specs(self, tenant_id: str, limit: int = 100) -> List[ProcedureSpec]:
        rows = self.db.query("SELECT * FROM specs WHERE tenant_id=? ORDER BY created_at DESC LIMIT ?",
                             (tenant_id, limit))
        return [ProcedureSpec.from_row(r) for r in rows]


SPECS = SpecStore(DB)


class ConversationStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, tenant_id: str, title: str) -> Dict[str, Any]:
        conversation_id = new_id("cnv")
        now = iso()
        self.db.execute(
            "INSERT INTO conversations(conversation_id, tenant_id, title, created_at, updated_at) VALUES(?,?,?,?,?)",
            (conversation_id, tenant_id, title[:300] or "New chat", now, now),
        )
        return {"conversation_id": conversation_id, "tenant_id": tenant_id, "title": title[:300] or "New chat",
                "created_at": now, "updated_at": now}

    def get(self, tenant_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.query_one("SELECT * FROM conversations WHERE conversation_id=? AND tenant_id=?",
                                (conversation_id, tenant_id))
        return {k: row[k] for k in row.keys()} if row else None

    def ensure(self, tenant_id: str, conversation_id: Optional[str], title: str) -> Dict[str, Any]:
        if conversation_id:
            existing = self.get(tenant_id, conversation_id)
            if existing:
                return existing
        return self.create(tenant_id, title)

    def list_conversations(self, tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM conversations WHERE tenant_id=? ORDER BY updated_at DESC LIMIT ?", (tenant_id, limit)
        )
        return [{k: r[k] for k in r.keys()} for r in rows]

    def delete(self, tenant_id: str, conversation_id: str) -> bool:
        existing = self.get(tenant_id, conversation_id)
        if not existing:
            return False
        self.db.execute("DELETE FROM conversations WHERE conversation_id=? AND tenant_id=?",
                        (conversation_id, tenant_id))
        return True

    def add_message(self, tenant_id: str, conversation_id: str, role: str, content: str,
                    meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        message_id = new_id("msg")
        now = iso()
        self.db.execute(
            "INSERT INTO messages(message_id, conversation_id, tenant_id, role, content, meta, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (message_id, conversation_id, tenant_id, role, content, jdump(meta or {}), now),
        )
        self.db.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=?", (now, conversation_id))
        return {"message_id": message_id, "conversation_id": conversation_id, "role": role, "content": content,
                "meta": meta or {}, "created_at": now}

    def messages(self, tenant_id: str, conversation_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM messages WHERE conversation_id=? AND tenant_id=? ORDER BY created_at ASC LIMIT ?",
            (conversation_id, tenant_id, limit),
        )
        return [{"message_id": r["message_id"], "role": r["role"], "content": r["content"],
                 "meta": jload(r["meta"], {}) or {}, "created_at": r["created_at"]} for r in rows]

    def rename(self, tenant_id: str, conversation_id: str, title: str) -> bool:
        if not self.get(tenant_id, conversation_id):
            return False
        self.db.execute("UPDATE conversations SET title=?, updated_at=? WHERE conversation_id=?",
                        (title[:300], iso(), conversation_id))
        return True


CONVERSATIONS = ConversationStore(DB)


class DeliberativeEngine:
    def __init__(self, model: ModelClient, memory: MemorySubsystem) -> None:
        self.model = model
        self.memory = memory

    def build_frame(self, run_id: str, spec: ProcedureSpec, state: ExecutionState,
                    observation: Observation, directives: List[str]) -> CognitionFrame:
        wm = self.memory.wm.derive(state, spec)
        seed_text = jdump({"wm": wm, "obs": observation.to_dict(), "directives": directives})
        base = hashed_embedding(seed_text, COGNITION_K * COGNITION_H)
        tokens: List[List[float]] = []
        for k in range(COGNITION_K):
            chunk = base[k * COGNITION_H : (k + 1) * COGNITION_H]
            if len(chunk) < COGNITION_H:
                chunk = list(chunk) + [0.0] * (COGNITION_H - len(chunk))
            tokens.append([round(float(v), 6) for v in chunk])
        open_goals = wm["open_subgoals"]
        subgoal = open_goals[0]["goal"] if open_goals else (state.next_intent or spec.objective[:200])
        blocked = 1.0 if wm["unresolved_blockers"] else 0.0
        gates = {
            "explore": clamp(0.85 - state.progress * 0.6 + blocked * 0.2, 0.0, 1.0),
            "exploit": clamp(0.15 + state.progress * 0.7, 0.0, 1.0),
            "verify": clamp(0.2 + state.progress * 0.75, 0.0, 1.0),
            "escalate": clamp(blocked * 0.8 + (0.3 if not observation.ok else 0.0), 0.0, 1.0),
            "consolidate": clamp(state.progress - 0.75, 0.0, 1.0),
        }
        directive_text = " | ".join(directives[:3]) if directives else (
            "Advance the highest-priority open subgoal with a verifiable action."
        )
        return CognitionFrame(
            frame_id=new_id("cog"),
            run_id=run_id,
            generated_at=time.time(),
            tokens=tokens,
            gates=gates,
            subgoal=str(subgoal)[:400],
            directive=directive_text[:1200],
            horizon=max(1, min(12, spec.max_steps - len([g for g in state.subgoals if isinstance(g, dict) and g.get("done")]))),
        )


DELIBERATOR = DeliberativeEngine(MODEL, MEMORY)


STEP_SYSTEM_PROMPT = (
    "You are the reasoning core of a stateless, long-horizon autonomous agent runtime.\n"
    "You receive ONLY: the immutable task specification P, the structured execution state SIGMA_t, the latest "
    "observation O_t, routed procedural skills, knowledge caveats, and a cognition frame. You never receive "
    "conversational history; SIGMA_t is the complete sufficient statistic.\n"
    "You may reason internally in multiple steps, but you MUST emit exactly one JSON object and nothing else.\n"
    "Required JSON shape:\n"
    "{\n"
    '  "state_delta": {"phase": "...", "progress": 0.0, "subgoals": [{"id":"g1","goal":"...","done":false,'
    '"depends_on":[]}], "facts": {}, "artifacts": {}, "blockers": [], "constraints_observed": [], '
    '"next_intent": "...", "scratch": {}, "metrics": {}, "skill_hints": []},\n'
    '  "action": {"tool": "<registered tool>", "arguments": {}, "rationale_digest": "one short sentence", '
    '"terminal": false, "final_answer": null}\n'
    "}\n"
    "Rules:\n"
    "1. state_delta uses dictionary-merge semantics. Include ONLY changed keys. Use JSON null to delete a key.\n"
    "2. Never restate unchanged state. Keep the delta minimal and O(1) in size.\n"
    "3. progress is a float in [0,1]. Set terminal true and provide final_answer only when success criteria are "
    "verifiably satisfied.\n"
    "4. Use the tool 'finish' with terminal true to end. Use 'noop' only to stabilize state.\n"
    "5. Do not emit markdown fences, comments, or any prose outside the single JSON object.\n"
    "6. Record durable, reusable procedures with record_skill and durable caveats with memory_write.\n"
    "7. If an observation reports an error twice for the same approach, change strategy rather than retrying.\n"
)


class StateTransitionEngine:
    def __init__(self, model: ModelClient, memory: MemorySubsystem) -> None:
        self.model = model
        self.memory = memory

    def build_prompt(self, spec: ProcedureSpec, state: ExecutionState, observation: Observation,
                     skills: List[Skill], caveats: List[Dict[str, Any]], frame: CognitionFrame,
                     directives: List[str], allowed_tools: Set[str], attempt: int,
                     validation_feedback: Optional[List[str]]) -> List[Dict[str, str]]:
        payload = {
            "P": spec.frozen_view(),
            "SIGMA_t": state.compact(),
            "WM": self.memory.wm.derive(state, spec),
            "O_t": observation.to_dict(),
            "routed_skills": [s.prompt_view() for s in skills],
            "knowledge_caveats": [
                {"slug": c["slug"], "title": c["title"], "excerpt": c["body"][-2400:]} for c in caveats
            ],
            "cognition_frame": frame.prompt_view(),
            "distilled_directives": directives[:4],
            "tool_catalog": TOOLS.catalog(allowed_tools),
            "attempt": attempt,
            "validation_feedback": validation_feedback or [],
        }
        return [
            {"role": "system", "content": STEP_SYSTEM_PROMPT},
            {"role": "user", "content": jdump(payload)},
        ]

    def decide(self, tenant: Tenant, spec: ProcedureSpec, state: ExecutionState, observation: Observation,
               skills: List[Skill], caveats: List[Dict[str, Any]], frame: CognitionFrame,
               directives: List[str], allowed_tools: Set[str]) -> Tuple[StepDecision, str, int]:
        feedback: Optional[List[str]] = None
        last_error = "no attempt executed"
        total_tokens = 0
        raw_text = ""
        for attempt in range(1, MAX_STEP_RETRIES + 1):
            messages = self.build_prompt(spec, state, observation, skills, caveats, frame, directives,
                                         allowed_tools, attempt, feedback)
            try:
                raw_text, tokens = self.model.complete(messages)
                total_tokens += tokens
            except Exception as exc:
                last_error = f"model invocation failed: {type(exc).__name__}: {exc}"
                feedback = [last_error]
                time.sleep(min(4.0, 0.6 * attempt))
                continue
            try:
                parsed = GrammarDecoder.extract_json(raw_text)
                delta = DeltaValidator.validate(parsed.get("state_delta", {}) or {}, allowed_tools)
                action = DeltaValidator.validate_action(parsed.get("action", {}) or {}, allowed_tools)
                decision = StepDecision(
                    delta=delta,
                    action=action,
                    reasoning_tokens=total_tokens,
                    raw_len=len(raw_text),
                )
                return decision, raw_text, total_tokens
            except (SchemaError, ValidationRejected) as exc:
                reasons = exc.reasons if isinstance(exc, ValidationRejected) else [str(exc)]
                last_error = "; ".join(reasons)
                feedback = reasons + [
                    "Your previous output was rejected by the deterministic validator. "
                    "Emit exactly one JSON object with keys state_delta and action. No prose."
                ]
        fallback_delta = {
            "blockers": (state.blockers + [f"state transition validation failure: {last_error[:300]}"])[:12],
            "next_intent": "Recover from schema validation failure with a minimal verifiable action.",
        }
        decision = StepDecision(
            delta=fallback_delta,
            action=ActionCommand(tool="noop", arguments={"reason": last_error[:400]},
                                 rationale_digest="validator fallback"),
            reasoning_tokens=total_tokens,
            raw_len=len(raw_text),
        )
        return decision, raw_text, total_tokens


TRANSITION = StateTransitionEngine(MODEL, MEMORY)


class OrchestratorGraph:
    def __init__(self) -> None:
        self.edges: Dict[NodeKind, List[NodeKind]] = {
            NodeKind.PERCEIVE: [NodeKind.DELIBERATE],
            NodeKind.DELIBERATE: [NodeKind.ACT],
            NodeKind.ACT: [NodeKind.VALIDATE],
            NodeKind.VALIDATE: [NodeKind.PERCEIVE, NodeKind.REFLECT],
            NodeKind.REFLECT: [NodeKind.CONSOLIDATE],
            NodeKind.CONSOLIDATE: [NodeKind.TERMINAL],
            NodeKind.TERMINAL: [],
        }

    def can_transition(self, src: NodeKind, dst: NodeKind) -> bool:
        return dst in self.edges.get(src, [])

    def describe(self) -> Dict[str, List[str]]:
        return {k.value: [v.value for v in vs] for k, vs in self.edges.items()}


GRAPH = OrchestratorGraph()


class AgentWorker(threading.Thread):
    def __init__(self, run_id: str, owner: str, supervisor: "Supervisor") -> None:
        super().__init__(name=f"agent-{run_id}", daemon=True)
        self.run_id = run_id
        self.owner = owner
        self.supervisor = supervisor
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self._frame: Optional[CognitionFrame] = None
        self._frame_lock = threading.RLock()
        self._system2: Optional[threading.Thread] = None
        self._distill_records: List[Dict[str, Any]] = []
        self._skill_attribution: Dict[int, str] = {}

    def request_stop(self) -> None:
        self.stop_event.set()

    def emit(self, kind: str, payload: Dict[str, Any]) -> None:
        message = {"type": kind, "run_id": self.run_id, "ts": iso(), **payload}
        BUS.publish_threadsafe(f"run:{self.run_id}", message)
        BUS.publish_threadsafe("*", message)

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            LOG.exception("worker crashed for run %s", self.run_id)
            RUNS.persist(self.run_id, status=RunState.FAILED, error=f"worker crash: {type(exc).__name__}: {exc}")
            self.emit("error", {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-4000:]})
        finally:
            self.stop_event.set()
            if self._system2 is not None:
                with contextlib.suppress(Exception):
                    self._system2.join(timeout=3.0)
            RUNS.release_lease(self.run_id, self.owner)
            self.supervisor.forget(self.run_id)
            self.emit("worker_exit", {})

    def _load(self) -> Tuple[RunRecord, ProcedureSpec, Tenant]:
        record = RUNS.get(self.run_id)
        if record is None:
            raise RuntimeError(f"run {self.run_id} not found")
        spec = SPECS.get(record.spec_id)
        if spec is None:
            raise RuntimeError(f"spec {record.spec_id} not found")
        tenant = TENANTS.by_id(record.tenant_id)
        if tenant is None:
            raise RuntimeError(f"tenant {record.tenant_id} not found")
        return record, spec, tenant

    def _system2_loop(self, spec: ProcedureSpec, run_id: str) -> None:
        period = 1.0 / max(0.05, SYSTEM2_HZ)
        while not self.stop_event.is_set():
            started = time.time()
            try:
                record = RUNS.get(run_id)
                if record is None:
                    return
                observation = Observation(
                    step=record.step,
                    source=str(record.last_observation.get("source", "bootstrap")),
                    ok=bool(record.last_observation.get("ok", True)),
                    payload=record.last_observation.get("payload", {}) or {},
                    error=record.last_observation.get("error"),
                )
                signature = PolicyPriorStore.signature(spec, record.state)
                directives = PRIORS.active_directives(record.tenant_id, signature) or \
                    PRIORS.global_directives(record.tenant_id)
                frame = DELIBERATOR.build_frame(run_id, spec, record.state, observation, directives)
                with self._frame_lock:
                    self._frame = frame
                self.emit("cognition", {"frame": frame.prompt_view()})
            except Exception as exc:
                LOG.warning("system2 loop error on %s: %s", run_id, exc)
            elapsed = time.time() - started
            self.stop_event.wait(max(0.0, period - elapsed))

    def _current_frame(self, spec: ProcedureSpec, record: RunRecord, observation: Observation) -> CognitionFrame:
        with self._frame_lock:
            frame = self._frame
        if frame is None or frame.staleness() > 30.0:
            signature = PolicyPriorStore.signature(spec, record.state)
            directives = PRIORS.active_directives(record.tenant_id, signature) or \
                PRIORS.global_directives(record.tenant_id)
            frame = DELIBERATOR.build_frame(self.run_id, spec, record.state, observation, directives)
            with self._frame_lock:
                self._frame = frame
        return frame

    def _execute(self) -> None:
        record, spec, tenant = self._load()
        sandbox = FileSystemSandbox(WORKSPACE_ROOT / tenant.tenant_id / self.run_id)
        allowed_tools = set(spec.allowed_tools) | {"finish", "noop"}
        checkpoint = CHECKPOINTS.latest(self.run_id)
        if checkpoint:
            record.state = ExecutionState.from_dict(checkpoint["state"])
            record.step = int(checkpoint["step"])
            self.emit("resumed", {"from_step": record.step, "checkpoint_id": checkpoint["checkpoint_id"]})
        RUNS.persist(self.run_id, status=RunState.RUNNING, node=NodeKind.PERCEIVE, state=record.state,
                     step=record.step, error="")
        LEDGER.append(tenant.tenant_id, "run_started", {"run_id": self.run_id, "step": record.step}, self.run_id)
        self.emit("status", {"status": RunState.RUNNING.value, "step": record.step})

        self._system2 = threading.Thread(target=self._system2_loop, args=(spec, self.run_id),
                                        name=f"system2-{self.run_id}", daemon=True)
        self._system2.start()

        observation = Observation(
            step=record.step,
            source=str(record.last_observation.get("source", "bootstrap")) or "bootstrap",
            ok=bool(record.last_observation.get("ok", True)),
            payload=record.last_observation.get("payload", {}) or {"note": "runtime initialized"},
            error=record.last_observation.get("error"),
        )
        period = 1.0 / max(1.0, SYSTEM1_HZ)
        terminal_answer = ""
        terminated = False
        failure_reason = ""

        while not self.stop_event.is_set():
            loop_started = time.time()
            live = RUNS.get(self.run_id)
            if live is None:
                failure_reason = "run record vanished"
                break
            if live.status == RunState.CANCELLED:
                self.emit("status", {"status": RunState.CANCELLED.value, "step": record.step})
                LEDGER.append(tenant.tenant_id, "run_cancelled", {"step": record.step}, self.run_id)
                return
            if live.status == RunState.PAUSED:
                self.emit("status", {"status": RunState.PAUSED.value, "step": record.step})
                self.stop_event.wait(1.0)
                continue
            if record.step >= spec.max_steps:
                failure_reason = f"step budget exhausted at {record.step}/{spec.max_steps}"
                break
            if record.tokens_used >= spec.token_budget:
                failure_reason = f"token budget exhausted ({record.tokens_used}/{spec.token_budget})"
                break

            RUNS.renew_lease(self.run_id, self.owner)
            step = record.step + 1
            pre_state = record.state
            pre_digest = stable_hash(pre_state.to_dict())

            RUNS.persist(self.run_id, node=NodeKind.DELIBERATE)
            frame = self._current_frame(spec, record, observation)
            skills = MEMORY.route_skills(tenant.tenant_id, pre_state, spec, limit=2)
            caveats = MEMORY.route_caveats(tenant.tenant_id, pre_state, spec, limit=2)
            signature = PolicyPriorStore.signature(spec, pre_state)
            directives = PRIORS.active_directives(tenant.tenant_id, signature) or \
                PRIORS.global_directives(tenant.tenant_id)

            self.emit("step_begin", {
                "step": step,
                "phase": pre_state.phase,
                "routed_skills": [s.name for s in skills],
                "caveats": [c["slug"] for c in caveats],
                "gates": {k: round(v, 3) for k, v in frame.gates.items()},
                "staleness_s": round(frame.staleness(), 3),
            })

            try:
                decision, raw_text, tokens = TRANSITION.decide(
                    tenant, spec, pre_state, observation, skills, caveats, frame, directives, allowed_tools
                )
            except Exception as exc:
                failure_reason = f"transition engine failure: {type(exc).__name__}: {exc}"
                break

            try:
                TENANTS.charge(tenant.tenant_id, tokens)
            except BudgetExceeded as exc:
                failure_reason = str(exc)
                break

            record.tokens_used += tokens
            self._distill_records.append({
                "step": step,
                "signature": signature,
                "student_prompt": jdump({"P": spec.frozen_view(), "SIGMA": pre_state.compact(1500)}),
                "student_completion": raw_text[:20000],
            })
            if len(self._distill_records) > 256:
                self._distill_records = self._distill_records[-256:]

            try:
                next_state = DeltaValidator.merge(pre_state, decision.delta)
            except Exception as exc:
                self.emit("validation_rollback", {"step": step, "reason": f"merge failure: {exc}"})
                next_state = pre_state

            action = decision.action
            RUNS.persist(self.run_id, node=NodeKind.ACT)
            self.emit("action", {"step": step, "tool": action.tool,
                                 "arguments": _safe_args(action.arguments),
                                 "rationale": action.rationale_digest,
                                 "terminal": action.terminal})

            ctx = ToolContext(tenant=tenant, spec=spec, run_id=self.run_id, step=step, sandbox=sandbox,
                              memory=MEMORY, state=next_state)
            if action.terminal or action.tool == "finish":
                terminal_answer = action.final_answer or str(action.arguments.get("final_answer", ""))
                observation = Observation(step=step, source="finish", ok=True,
                                          payload={"final_answer": terminal_answer[:4000], "terminal": True})
                terminated = True
            else:
                observation = TOOLS.execute(ctx, action)

            selected_skill = skills[0].skill_id if skills else None
            if selected_skill:
                self._skill_attribution[step] = selected_skill
                MEMORY.em.record_outcome(selected_skill, bool(observation.ok))

            RUNS.persist(self.run_id, node=NodeKind.VALIDATE)
            trace = ExecutionTrace(
                trace_id=new_id("trc"),
                run_id=self.run_id,
                step=step,
                pre_state_digest=pre_digest,
                selected_skill=selected_skill,
                tool=action.tool,
                outcome="ok" if observation.ok else "error",
                delta_digest=stable_hash(decision.delta),
                post_state_digest=stable_hash(next_state.to_dict()),
                receipt={
                    "arguments": _safe_args(action.arguments),
                    "payload": _clip_payload(observation.payload),
                    "error": observation.error,
                    "latency_ms": observation.latency_ms,
                    "tokens": tokens,
                    "rationale_digest": action.rationale_digest,
                },
            )
            MEMORY.record_trace(trace)
            LEDGER.append(tenant.tenant_id, "step_receipt", {
                "step": step, "tool": action.tool, "ok": observation.ok,
                "delta_digest": trace.delta_digest, "post_digest": trace.post_state_digest,
            }, self.run_id)

            if not observation.ok and observation.error:
                blockers = list(next_state.blockers)
                marker = f"step {step} {action.tool}: {observation.error[:220]}"
                if marker not in blockers:
                    blockers.append(marker)
                next_state.blockers = blockers[-12:]

            record.state = next_state
            record.step = step
            RUNS.persist(self.run_id, step=step, state=next_state, observation=observation,
                         tokens_used=record.tokens_used, node=NodeKind.PERCEIVE)
            if step % max(1, CHECKPOINT_EVERY) == 0:
                CHECKPOINTS.save(self.run_id, step, NodeKind.PERCEIVE, next_state, observation,
                                 RunState.RUNNING, record.tokens_used)

            self.emit("step_end", {
                "step": step,
                "tool": action.tool,
                "ok": observation.ok,
                "error": observation.error,
                "delta_keys": sorted(decision.delta.keys()),
                "state": next_state.compact(3000),
                "tokens_used": record.tokens_used,
                "observation": _clip_payload(observation.payload),
            })

            if terminated:
                break

            elapsed = time.time() - loop_started
            if elapsed < period:
                self.stop_event.wait(period - elapsed)

        if self.stop_event.is_set() and not terminated and not failure_reason:
            RUNS.persist(self.run_id, status=RunState.PAUSED, state=record.state)
            self.emit("status", {"status": RunState.PAUSED.value, "step": record.step})
            return

        RUNS.persist(self.run_id, node=NodeKind.REFLECT)
        verdict = Verifier.verify(spec, record.state, sandbox, terminal_answer)
        if failure_reason and verdict["passed"]:
            verdict["passed"] = False
            verdict["checks"].append({"criterion": "runtime_budget", "passed": False, "detail": failure_reason})
        self.emit("verification", {"verdict": verdict, "reason": failure_reason})

        traces = MEMORY.run_traces(self.run_id)
        patch = REFLECTOR.build_patch(tenant, spec, self.run_id, verdict, record.state, traces)
        self.emit("reflection", {"patch": patch.to_dict()})

        distill = DISTILLER.distill(tenant.tenant_id, self.run_id, patch, self._distill_records)
        self.emit("distillation", distill)

        RUNS.persist(self.run_id, node=NodeKind.CONSOLIDATE)
        consolidation = META.consolidate(tenant, spec, self.run_id, patch, record.state, traces)
        self.emit("consolidation", consolidation)

        final_status = RunState.SUCCEEDED if verdict["passed"] else RunState.FAILED
        answer = terminal_answer or self._synthesize_answer(spec, record.state, verdict, patch)
        RUNS.persist(self.run_id, status=final_status, node=NodeKind.TERMINAL, final_answer=answer,
                     error=failure_reason, state=record.state)
        CHECKPOINTS.save(self.run_id, record.step, NodeKind.TERMINAL, record.state,
                         Observation(step=record.step, source="terminal", ok=verdict["passed"],
                                     payload={"final_answer": answer[:4000]}),
                         final_status, record.tokens_used)
        CHECKPOINTS.prune(self.run_id)
        LEDGER.append(tenant.tenant_id, "run_finished", {
            "status": final_status.value, "steps": record.step, "tokens": record.tokens_used,
            "verdict": verdict["passed"],
        }, self.run_id)

        live = RUNS.get(self.run_id)
        if live and live.conversation_id:
            CONVERSATIONS.add_message(
                tenant.tenant_id, live.conversation_id, "assistant", answer,
                {"run_id": self.run_id, "status": final_status.value, "steps": record.step,
                 "tokens": record.tokens_used, "verified": verdict["passed"]},
            )
        self.emit("final", {"status": final_status.value, "final_answer": answer, "steps": record.step,
                            "tokens_used": record.tokens_used, "verified": verdict["passed"]})

    def _synthesize_answer(self, spec: ProcedureSpec, state: ExecutionState, verdict: Dict[str, Any],
                           patch: ReflectionPatch) -> str:
        lines: List[str] = []
        lines.append(f"Objective: {spec.objective[:600]}")
        lines.append(f"Terminal phase: {state.phase} (progress {round(state.progress * 100)}%)")
        done = [g for g in state.subgoals if isinstance(g, dict) and g.get("done")]
        pending = [g for g in state.subgoals if isinstance(g, dict) and not g.get("done")]
        if done:
            lines.append("Completed subgoals: " + "; ".join(str(g.get("goal", ""))[:160] for g in done[:8]))
        if pending:
            lines.append("Outstanding subgoals: " + "; ".join(str(g.get("goal", ""))[:160] for g in pending[:8]))
        if state.artifacts:
            lines.append("Artifacts: " + ", ".join(sorted(state.artifacts.keys())[:12]))
        failed = [c["criterion"] for c in verdict.get("checks", []) if not c.get("passed")]
        if failed:
            lines.append("Unsatisfied criteria: " + "; ".join(str(f)[:160] for f in failed[:6]))
        if state.blockers:
            lines.append("Blockers: " + "; ".join(b[:160] for b in state.blockers[:5]))
        lines.append(f"Diagnosis: {patch.root_cause[:600]}")
        if patch.guidance:
            lines.append(f"Next-run guidance: {patch.guidance[:600]}")
        return "\n".join(lines)


def _safe_args(args: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in (args or {}).items():
        text = value if isinstance(value, str) else jdump(value)
        if len(text) > 2000:
            text = text[:2000] + "...[clipped]"
        out[key] = OutputClassifier.redact(text) if isinstance(value, str) else text
    return out


def _clip_payload(payload: Dict[str, Any], limit: int = 6000) -> Dict[str, Any]:
    blob = jdump(payload or {})
    if len(blob) <= limit:
        return payload or {}
    out: Dict[str, Any] = {}
    used = 0
    for key in sorted((payload or {}).keys()):
        chunk = jdump({key: payload[key]})
        if used + len(chunk) > limit:
            out["__clipped__"] = True
            break
        out[key] = payload[key]
        used += len(chunk)
    return out


class Supervisor:
    def __init__(self) -> None
        self.owner = f"{os.getpid()}-{secrets.token_hex(4)}"
        self._workers: Dict[str, AgentWorker] = {}
        self._lock = threading.RLock()
        self._reaper: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    def start_background(self) -> None:
        if self._reaper is None or not self._reaper.is_alive():
            self._reaper = threading.Thread(target=self._reap_loop, name="supervisor-reaper", daemon=True)
            self._reaper.start()

    def shutdown(self) -> None:
        self._shutdown.set()
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.request_stop()
        for worker in workers:
            with contextlib.suppress(Exception):
                worker.join(timeout=5.0)

    def _reap_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                self.recover_orphans()
                with self._lock:
                    dead = [rid for rid, w in self._workers.items() if not w.is_alive()]
                    for rid in dead:
                        self._workers.pop(rid, None)
            except Exception as exc:
                LOG.warning("supervisor reaper error: %s", exc)
            self._shutdown.wait(10.0)

    def recover_orphans(self) -> List[str]:
        recovered: List[str] = []
        for record in RUNS.resumable():
            with self._lock:
                if record.run_id in self._workers:
                    continue
            if RUNS.acquire_lease(record.run_id, self.owner):
                RUNS.persist(record.run_id, status=RunState.RECOVERING)
                LOG.info("recovering orphaned run %s from step %s", record.run_id, record.step)
                self._spawn(record.run_id)
                recovered.append(record.run_id)
        return recovered

    def _spawn(self, run_id: str) -> AgentWorker:
        worker = AgentWorker(run_id, self.owner, self)
        with self._lock:
            self._workers[run_id] = worker
        worker.start()
        return worker

    def launch(self, run_id: str) -> bool:
        with self._lock:
            existing = self._workers.get(run_id)
            if existing is not None and existing.is_alive():
                return False
        if not RUNS.acquire_lease(run_id, self.owner):
            return False
        self._spawn(run_id)
        return True

    def pause(self, run_id: str) -> bool:
        record = RUNS.get(run_id)
        if record is None or record.status not in (RunState.RUNNING, RunState.PENDING, RunState.RECOVERING):
            return False
        RUNS.persist(run_id, status=RunState.PAUSED)
        return True

    def resume(self, run_id: str) -> bool:
        record = RUNS.get(run_id)
        if record is None:
            return False
        if record.status in (RunState.SUCCEEDED, RunState.FAILED):
            return False
        RUNS.persist(run_id, status=RunState.RUNNING)
        with self._lock:
            worker = self._workers.get(run_id)
        if worker is not None and worker.is_alive():
            return True
        return self.launch(run_id)

    def cancel(self, run_id: str) -> bool:
        record = RUNS.get(run_id)
        if record is None:
            return False
        RUNS.request_cancel(run_id)
        with self._lock:
            worker = self._workers.get(run_id)
        if worker is not None:
            worker.request_stop()
        return True

    def forget(self, run_id: str) -> None:
        with self._lock:
            self._workers.pop(run_id, None)

    def active(self) -> List[str]:
        with self._lock:
            return [rid for rid, w in self._workers.items() if w.is_alive()]


SUPERVISOR = Supervisor()


class CreateConversationRequest(BaseModel):
    title: str = Field(default="New chat", max_length=300)


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=200000)
    conversation_id: Optional[str] = None
    stream: bool = True
    autonomous: bool = False
    title: Optional[str] = Field(default=None, max_length=300)
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    allowed_tools: List[str] = Field(default_factory=list)
    max_steps: int = Field(default=DEFAULT_MAX_STEPS, ge=1, le=5000)
    token_budget: int = Field(default=DEFAULT_TOKEN_BUDGET, ge=1000)
    verifier_program: Optional[str] = None

    @field_validator("constraints", "success_criteria", "allowed_tools")
    @classmethod
    def _cap_lists(cls, value: List[str]) -> List[str]:
        return [str(v)[:600] for v in value][:32]


class CreateRunRequest(BaseModel):
    title: str = Field(default="Autonomous objective", max_length=300)
    objective: str = Field(min_length=1, max_length=20000)
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    allowed_tools: List[str] = Field(default_factory=list)
    max_steps: int = Field(default=DEFAULT_MAX_STEPS, ge=1, le=5000)
    token_budget: int = Field(default=DEFAULT_TOKEN_BUDGET, ge=1000)
    verifier_program: Optional[str] = None
    conversation_id: Optional[str] = None
    autostart: bool = True


class SkillUpsertRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=4000)
    when_to_use: str = Field(default="", max_length=2000)
    procedure: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)


class WikiUpsertRequest(BaseModel):
    slug: str = Field(min_length=1, max_length=120)
    title: str = Field(default="", max_length=300)
    body: str = Field(default="", max_length=400000)


class DiagnosticRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    payload: Dict[str, Any] = Field(default_factory=dict)
    expectation: Dict[str, Any] = Field(default_factory=dict)


class TenantCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    api_key: str = Field(min_length=8, max_length=256)
    token_budget: int = Field(default=DEFAULT_TOKEN_BUDGET, ge=1000)
    allowed_risk: str = Field(default="guarded")


def extract_api_key(request: Request, authorization: Optional[str], x_api_key: Optional[str]) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    if x_api_key:
        return x_api_key.strip()
    query_key = request.query_params.get("api_key")
    if query_key:
        return query_key.strip()
    return os.environ.get("AGENT_DEFAULT_API_KEY", "local-dev-key")


async def require_tenant(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Tenant:
    key = extract_api_key(request, authorization, x_api_key)
    tenant = TENANTS.by_api_key(key)
    if tenant is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")
    return tenant


async def require_admin(x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")) -> bool:
    if not ADMIN_TOKEN:
        return True
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin token required")
    return True


app = FastAPI(title="Autonomous Agent Runtime", version="1.0.0", docs_url="/api/docs", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("AGENT_CORS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _on_startup() -> None:
    BUS.bind_loop(asyncio.get_running_loop())
    SUPERVISOR.start_background()
    _seed_defaults()
    LOG.info("runtime online | model=%s | tools=%d | git=%s", MODEL_NAME, len(TOOLS.names()), GIT_ENABLED)


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    SUPERVISOR.shutdown()
    LOG.info("runtime shutdown complete")


def _seed_defaults() -> None:
    for tenant in TENANTS.list_all():
        if not MEMORY.em.list_skills(tenant.tenant_id, include_quarantined=True):
            MEMORY.em.upsert(
                tenant_id=tenant.tenant_id,
                name="decompose_and_verify",
                description="Baseline procedure: decompose the objective into atomic verifiable subgoals, "
                            "execute one at a time, and assert each success criterion before terminating.",
                trigger_signature="bootstrap phase, no subgoals present, or objective is broad and underspecified",
                procedure=[
                    "Restate the objective as 3-7 atomic subgoals with explicit completion predicates.",
                    "Persist the subgoals into state_delta.subgoals with stable ids and dependencies.",
                    "Select the first subgoal whose dependencies are satisfied.",
                    "Execute exactly one tool call that materially advances that subgoal.",
                    "Record verified outcomes into state_delta.facts and artifacts.",
                    "Mark the subgoal done only when its completion predicate is objectively satisfied.",
                    "Before terminal finish, re-check every declared success criterion with a read or check tool.",
                ],
                tools=["read_file", "write_file", "check_lines", "list_dir", "reason", "finish"],
                reason="runtime bootstrap seed",
            )
            MEMORY.em.upsert(
                tenant_id=tenant.tenant_id,
                name="recover_from_tool_error",
                description="Deterministic recovery ladder when a tool returns an error, preventing identical retries.",
                trigger_signature="latest observation ok=false, or blockers list is non-empty",
                procedure=[
                    "Read the observation error verbatim and classify it: missing input, bad path, permission, or timeout.",
                    "For missing or bad paths, list_dir the parent directory before retrying.",
                    "For permission errors, choose a lower-risk tool that achieves the same effect.",
                    "For timeouts, reduce the work unit size and retry once.",
                    "If the same tool errors twice, change strategy and record the caveat with memory_write.",
                ],
                tools=["list_dir", "read_file", "search_files", "memory_write", "reason"],
                reason="runtime bootstrap seed",
            )
        if not GATE.diagnostics(tenant.tenant_id):
            GATE.add_diagnostic(
                tenant.tenant_id,
                "skill_declares_verification",
                {"query": "verify success criteria before finishing the task"},
                {"must_include": ["verif"], "must_not_include": ["ignore previous instructions"],
                 "min_relevance": 0.02},
            )
            GATE.add_diagnostic(
                tenant.tenant_id,
                "skill_has_bounded_procedure",
                {"query": "ordered atomic steps with explicit completion predicates"},
                {"must_include": [], "must_not_include": ["rm -rf /"], "min_relevance": 0.0},
            )
        if not MEMORY.wiki.list_pages(tenant.tenant_id):
            MEMORY.wiki.upsert(
                tenant.tenant_id,
                "runtime-playbook",
                "Runtime Playbook",
                "## Invariants\n"
                "- The execution state is the sole sufficient statistic; no conversational history is replayed.\n"
                "- Every state delta is validated deterministically before being merged.\n"
                "- Identical failing tool calls must never be retried more than once.\n"
                "- Success criteria are asserted with read or check tools before terminating.\n",
                reason="bootstrap",
            )


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "time": iso(),
        "model": MODEL_NAME,
        "model_configured": MODEL.available,
        "tools": TOOLS.names(),
        "graph": GRAPH.describe(),
        "active_runs": SUPERVISOR.active(),
        "system1_hz": SYSTEM1_HZ,
        "system2_hz": SYSTEM2_HZ,
        "git_versioning": GIT_ENABLED,
        "ledger": LEDGER.verify_chain(500),
    }


@app.get("/api/config")
async def config(tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    return {
        "tenant": {"tenant_id": tenant.tenant_id, "name": tenant.name,
                   "token_budget": tenant.token_budget, "tokens_used": tenant.tokens_used,
                   "allowed_risk": tenant.allowed_risk.value},
        "model": MODEL_NAME,
        "defaults": {"max_steps": DEFAULT_MAX_STEPS, "token_budget": DEFAULT_TOKEN_BUDGET},
        "tools": TOOLS.catalog(),
    }


@app.get("/api/conversations")
async def list_conversations(tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    return {"conversations": CONVERSATIONS.list_conversations(tenant.tenant_id)}


@app.post("/api/conversations")
async def create_conversation(body: CreateConversationRequest,
                              tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    return CONVERSATIONS.create(tenant.tenant_id, body.title)


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    conv = CONVERSATIONS.get(tenant.tenant_id, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    conv["messages"] = CONVERSATIONS.messages(tenant.tenant_id, conversation_id)
    conv["runs"] = [r.to_dict() for r in RUNS.list_runs(tenant.tenant_id, limit=200)
                    if r.conversation_id == conversation_id]
    return conv


@app.patch("/api/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, body: RenameConversationRequest,
                              tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    if not CONVERSATIONS.rename(tenant.tenant_id, conversation_id, body.title):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"ok": True, "conversation_id": conversation_id, "title": body.title}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    if not CONVERSATIONS.delete(tenant.tenant_id, conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"ok": True}


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {jdump(data)}\n\n"


@app.post("/api/chat")
async def chat(body: ChatRequest, tenant: Tenant = Depends(require_tenant)) -> Response:
    conv = CONVERSATIONS.ensure(tenant.tenant_id, body.conversation_id,
                                body.title or body.message[:80] or "New chat")
    conversation_id = conv["conversation_id"]
    CONVERSATIONS.add_message(tenant.tenant_id, conversation_id, "user", body.message, {"autonomous": body.autonomous})

    if body.autonomous:
        spec = SPECS.create(
            tenant_id=tenant.tenant_id,
            title=body.title or body.message[:120],
            objective=body.message,
            constraints=body.constraints,
            success_criteria=body.success_criteria,
            allowed_tools=body.allowed_tools,
            max_steps=body.max_steps,
            token_budget=body.token_budget,
            verifier_program=body.verifier_program,
        )
        record = RUNS.create(tenant.tenant_id, spec, conversation_id)
        SUPERVISOR.launch(record.run_id)
        return JSONResponse({
            "mode": "autonomous",
            "conversation_id": conversation_id,
            "run_id": record.run_id,
            "spec_id": spec.spec_id,
            "stream_url": f"/api/runs/{record.run_id}/events",
            "websocket_url": f"/ws/runs/{record.run_id}",
        })

    if not MODEL.available:
        raise HTTPException(status_code=503, detail="MODULAR_API_KEY is not configured on the server")

    history = CONVERSATIONS.messages(tenant.tenant_id, conversation_id, limit=40)
    messages: List[Dict[str, str]] = [{
        "role": "system",
        "content": "You are a helpful assistant. Be concise and accurate. Answer in plain text without markdown "
                   "unless asked.",
    }]
    for msg in history[-24:]:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"][:24000]})

    if not body.stream:
        try:
            text, tokens = MODEL.complete(messages)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"model error: {exc}") from exc
        with contextlib.suppress(BudgetExceeded, AuthorizationDenied):
            TENANTS.charge(tenant.tenant_id, tokens)
        verdict = OutputClassifier.classify(text)
        safe = text if verdict["allowed"] else OutputClassifier.redact(text)
        stored = CONVERSATIONS.add_message(tenant.tenant_id, conversation_id, "assistant", safe,
                                           {"tokens": tokens, "flags": verdict["flags"]})
        return JSONResponse({"mode": "chat", "conversation_id": conversation_id, "message": stored,
                             "tokens": tokens, "flags": verdict["flags"]})

    async def generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        accumulated: List[str] = []
        usage_holder = {"tokens": 0}

        def producer() -> None:
            try:
                for piece, total in MODEL.stream(messages):
                    if total is not None:
                        usage_holder["tokens"] = total
                    if piece:
                        accumulated.append(piece)
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "delta", "content": piece})
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "__eof__"})

        threading.Thread(target=producer, name="chat-stream", daemon=True).start()
        yield _sse("open", {"conversation_id": conversation_id})
        while True:
            item = await queue.get()
            if item.get("type") == "__eof__":
                break
            if item.get("type") == "error":
                yield _sse("error", item)
                break
            yield _sse("delta", item)
        full = "".join(accumulated)
        tokens = usage_holder["tokens"] or approx_tokens(full)
        with contextlib.suppress(BudgetExceeded, AuthorizationDenied):
            TENANTS.charge(tenant.tenant_id, tokens)
        verdict = OutputClassifier.classify(full)
        safe = full if verdict["allowed"] else OutputClassifier.redact(full)
        stored = CONVERSATIONS.add_message(tenant.tenant_id, conversation_id, "assistant", safe,
                                           {"tokens": tokens, "flags": verdict["flags"]})
        yield _sse("done", {"conversation_id": conversation_id, "message": stored, "tokens": tokens,
                            "flags": verdict["flags"]})

    return StreamingResponse(generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


@app.post("/api/runs")
async def create_run(body: CreateRunRequest, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    conversation_id = ""
    if body.conversation_id:
        conv = CONVERSATIONS.get(tenant.tenant_id, body.conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        conversation_id = conv["conversation_id"]
    spec = SPECS.create(
        tenant_id=tenant.tenant_id,
        title=body.title,
        objective=body.objective,
        constraints=body.constraints,
        success_criteria=body.success_criteria,
        allowed_tools=body.allowed_tools,
        max_steps=body.max_steps,
        token_budget=body.token_budget,
        verifier_program=body.verifier_program,
    )
    record = RUNS.create(tenant.tenant_id, spec, conversation_id)
    launched = SUPERVISOR.launch(record.run_id) if body.autostart else False
    return {
        "run_id": record.run_id,
        "spec_id": spec.spec_id,
        "status": RUNS.get(record.run_id).status.value if RUNS.get(record.run_id) else RunState.PENDING.value,
        "launched": launched,
        "stream_url": f"/api/runs/{record.run_id}/events",
        "websocket_url": f"/ws/runs/{record.run_id}",
    }


@app.get("/api/runs")
async def list_runs(status_filter: Optional[str] = Query(default=None, alias="status"),
                    limit: int = Query(default=100, ge=1, le=500),
                    tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    runs = RUNS.list_runs(tenant.tenant_id, status_filter, limit)
    return {"runs": [r.to_dict() for r in runs], "active": SUPERVISOR.active()}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    record = RUNS.get_scoped(tenant.tenant_id, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    spec = SPECS.get(record.spec_id)
    payload = record.to_dict()
    payload["spec"] = spec.frozen_view() if spec else None
    payload["checkpoint"] = CHECKPOINTS.latest(run_id)
    payload["traces"] = MEMORY.run_traces(run_id, 200)
    reflection = REFLECTOR.latest(run_id)
    payload["reflection"] = reflection.to_dict() if reflection else None
    return payload


@app.get("/api/runs/{run_id}/checkpoints")
async def run_checkpoints(run_id: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    if RUNS.get_scoped(tenant.tenant_id, run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"checkpoints": CHECKPOINTS.history(run_id), "latest": CHECKPOINTS.latest(run_id)}


@app.get("/api/runs/{run_id}/traces")
async def run_traces(run_id: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    if RUNS.get_scoped(tenant.tenant_id, run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"traces": MEMORY.run_traces(run_id, 500)}


@app.post("/api/runs/{run_id}/pause")
async def pause_run(run_id: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    if RUNS.get_scoped(tenant.tenant_id, run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"ok": SUPERVISOR.pause(run_id)}


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    if RUNS.get_scoped(tenant.tenant_id, run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"ok": SUPERVISOR.resume(run_id)}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    if RUNS.get_scoped(tenant.tenant_id, run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"ok": SUPERVISOR.cancel(run_id)}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, request: Request, tenant: Tenant = Depends(require_tenant)) -> StreamingResponse:
    record = RUNS.get_scoped(tenant.tenant_id, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    topic = f"run:{run_id}"
    queue = await BUS.subscribe(topic)

    async def generator() -> AsyncGenerator[str, None]:
        try:
            yield _sse("snapshot", record.to_dict())
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    live = RUNS.get(run_id)
                    yield _sse("heartbeat", {"ts": iso(), "status": live.status.value if live else "unknown"})
                    if live and live.status in (RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED):
                        yield _sse("closed", {"status": live.status.value})
                        break
                    continue
                yield _sse(message.get("type", "message"), message)
                if message.get("type") == "final":
                    yield _sse("closed", {"status": message.get("status")})
                    break
        finally:
            await BUS.unsubscribe(topic, queue)

    return StreamingResponse(generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })


@app.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    key = websocket.query_params.get("api_key") or os.environ.get("AGENT_DEFAULT_API_KEY", "local-dev-key")
    tenant = TENANTS.by_api_key(key)
    if tenant is None:
        await websocket.send_text(jdump({"type": "error", "error": "unauthorized"}))
        await websocket.close(code=4401)
        return
    record = RUNS.get_scoped(tenant.tenant_id, run_id)
    if record is None:
        await websocket.send_text(jdump({"type": "error", "error": "run not found"}))
        await websocket.close(code=4404)
        return
    topic = f"run:{run_id}"
    queue = await BUS.subscribe(topic)
    await websocket.send_text(jdump({"type": "snapshot", **record.to_dict()}))

    async def pump() -> None:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                live = RUNS.get(run_id)
                await websocket.send_text(jdump({"type": "heartbeat", "ts": iso(),
                                                 "status": live.status.value if live else "unknown"}))
                continue
            await websocket.send_text(jdump(message))
            if message.get("type") == "final":
                return

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            done, _pending = await asyncio.wait(
                {pump_task, asyncio.create_task(websocket.receive_text())},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pump_task in done:
                break
            for task in done:
                if task is pump_task:
                    continue
                try:
                    raw = task.result()
                except (WebSocketDisconnect, RuntimeError):
                    raise WebSocketDisconnect(code=1000)
                command = jload(raw, {}) or {}
                action = str(command.get("action", ""))
                if action == "pause":
                    await websocket.send_text(jdump({"type": "ack", "action": action, "ok": SUPERVISOR.pause(run_id)}))
                elif action == "resume":
                    await websocket.send_text(jdump({"type": "ack", "action": action, "ok": SUPERVISOR.resume(run_id)}))
                elif action == "cancel":
                    await websocket.send_text(jdump({"type": "ack", "action": action, "ok": SUPERVISOR.cancel(run_id)}))
                elif action == "ping":
                    await websocket.send_text(jdump({"type": "pong", "ts": iso()}))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        LOG.warning("websocket error on %s: %s", run_id, exc)
    finally:
        pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump_task
        await BUS.unsubscribe(topic, queue)
        with contextlib.suppress(Exception):
            await websocket.close()


@app.get("/api/skills")
async def list_skills(include_quarantined: bool = Query(default=False),
                      tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    skills = MEMORY.em.list_skills(tenant.tenant_id, include_quarantined)
    return {"skills": [{
        **s.prompt_view(),
        "description": s.description,
        "success_count": s.success_count,
        "failure_count": s.failure_count,
        "quarantined": bool(s.quarantined),
        "updated_at": s.updated_at,
    } for s in skills]}


@app.post("/api/skills")
async def upsert_skill(body: SkillUpsertRequest, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    registered = set(TOOLS.names())
    tools = [t for t in body.tools if t in registered]
    skill = MEMORY.em.upsert(
        tenant_id=tenant.tenant_id,
        name=re.sub(r"[^a-zA-Z0-9_]+", "_", body.name).strip("_").lower()[:48] or "skill",
        description=body.description,
        trigger_signature=body.when_to_use,
        procedure=[str(p)[:400] for p in body.procedure][:32],
        tools=tools[:16],
        reason="operator upsert",
    )
    MEMORY.em._write_disk(skill)
    return {"skill": skill.prompt_view()}


@app.post("/api/skills/{skill_id}/quarantine")
async def quarantine_skill(skill_id: str, enabled: bool = Query(default=True),
                           tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    skill = MEMORY.em.get(skill_id)
    if skill is None or skill.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="skill not found")
    MEMORY.em.set_quarantine(skill_id, enabled)
    return {"ok": True, "skill_id": skill_id, "quarantined": enabled}


@app.post("/api/skills/{skill_id}/rollback")
async def rollback_skill(skill_id: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    skill = MEMORY.em.get(skill_id)
    if skill is None or skill.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="skill not found")
    ok = MEMORY.em.rollback(skill_id)
    return {"ok": ok, "skill": (MEMORY.em.get(skill_id).prompt_view() if MEMORY.em.get(skill_id) else None)}


@app.get("/api/skills/{skill_id}/evaluate")
async def evaluate_skill(skill_id: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    skill = MEMORY.em.get(skill_id)
    if skill is None or skill.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="skill not found")
    return GATE.evaluate_skill(skill, GATE.diagnostics(tenant.tenant_id))


@app.get("/api/wiki")
async def list_wiki(tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    return {"pages": MEMORY.wiki.list_pages(tenant.tenant_id), "diff": MEMORY.wiki.diff(1)[:8000]}


@app.get("/api/wiki/{slug}")
async def get_wiki(slug: str, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    page = MEMORY.wiki.get(tenant.tenant_id, slug)
    if page is None:
        raise HTTPException(status_code=404, detail="page not found")
    page.pop("embedding", None)
    return page


@app.post("/api/wiki")
async def upsert_wiki(body: WikiUpsertRequest, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    return MEMORY.wiki.upsert(tenant.tenant_id, body.slug, body.title or body.slug, body.body, reason="operator")


@app.get("/api/memory/search")
async def memory_search(q: str = Query(min_length=1), limit: int = Query(default=5, ge=1, le=25),
                        tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    skills = MEMORY.em.retrieve(tenant.tenant_id, q, limit)
    pages = MEMORY.wiki.search(tenant.tenant_id, q, limit)
    return {
        "query": q,
        "skills": [s.prompt_view() for s in skills],
        "wiki": [{"slug": p["slug"], "title": p["title"], "excerpt": p["body"][:1200]} for p in pages],
    }


@app.get("/api/diagnostics")
async def list_diagnostics(tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    return {"diagnostics": GATE.diagnostics(tenant.tenant_id)}


@app.post("/api/diagnostics")
async def add_diagnostic(body: DiagnosticRequest, tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    task_id = GATE.add_diagnostic(tenant.tenant_id, body.name, body.payload, body.expectation)
    return {"task_id": task_id}


@app.get("/api/distillation")
async def distillation_stats(limit: int = Query(default=100, ge=1, le=1000),
                             tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    rows = DB.query(
        "SELECT sample_id, run_id, step, reverse_kl, weight, created_at FROM distillation_samples"
        " WHERE tenant_id=? ORDER BY created_at DESC LIMIT ?",
        (tenant.tenant_id, limit),
    )
    samples = [{k: r[k] for k in r.keys()} for r in rows]
    agg = DB.query_one(
        "SELECT COUNT(*) AS n, AVG(reverse_kl) AS mean_kl, MAX(reverse_kl) AS max_kl FROM distillation_samples"
        " WHERE tenant_id=?",
        (tenant.tenant_id,),
    )
    priors = DB.query(
        "SELECT signature, directive, logit, updates, updated_at FROM policy_priors WHERE tenant_id=?"
        " ORDER BY logit DESC LIMIT 50",
        (tenant.tenant_id,),
    )
    return {
        "samples": samples,
        "summary": {
            "count": int(agg["n"]) if agg else 0,
            "mean_reverse_kl": round(float(agg["mean_kl"] or 0.0), 8) if agg else 0.0,
            "max_reverse_kl": round(float(agg["max_kl"] or 0.0), 8) if agg else 0.0,
        },
        "priors": [{k: r[k] for k in r.keys()} for r in priors],
    }


@app.get("/api/ledger")
async def ledger(limit: int = Query(default=200, ge=1, le=2000),
                 run_id: Optional[str] = Query(default=None),
                 tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    if run_id:
        rows = DB.query(
            "SELECT event_id, kind, payload, hash, created_at FROM raw_events WHERE tenant_id=? AND run_id=?"
            " ORDER BY rowid DESC LIMIT ?",
            (tenant.tenant_id, run_id, limit),
        )
    else:
        rows = DB.query(
            "SELECT event_id, kind, payload, hash, created_at FROM raw_events WHERE tenant_id=?"
            " ORDER BY rowid DESC LIMIT ?",
            (tenant.tenant_id, limit),
        )
    return {
        "events": [{"event_id": r["event_id"], "kind": r["kind"], "payload": jload(r["payload"], {}),
                    "hash": r["hash"], "created_at": r["created_at"]} for r in rows],
        "integrity": LEDGER.verify_chain(1000),
    }


@app.get("/api/audit")
async def audit(limit: int = Query(default=200, ge=1, le=2000),
                tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    rows = DB.query(
        "SELECT audit_id, actor, action, resource, allowed, detail, created_at FROM audit_log WHERE tenant_id=?"
        " ORDER BY rowid DESC LIMIT ?",
        (tenant.tenant_id, limit),
    )
    return {"audit": [{k: r[k] for k in r.keys()} for r in rows]}


@app.get("/api/workspace/{run_id}")
async def workspace_listing(run_id: str, path: str = Query(default="."),
                            tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    if RUNS.get_scoped(tenant.tenant_id, run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    sandbox = FileSystemSandbox(WORKSPACE_ROOT / tenant.tenant_id / run_id)
    try:
        return sandbox.list_dir(path, 3)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/workspace/{run_id}/file")
async def workspace_file(run_id: str, path: str = Query(min_length=1),
                         tenant: Tenant = Depends(require_tenant)) -> Dict[str, Any]:
    if RUNS.get_scoped(tenant.tenant_id, run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    sandbox = FileSystemSandbox(WORKSPACE_ROOT / tenant.tenant_id / run_id)
    try:
        result = sandbox.read_file(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result["content"] = OutputClassifier.redact(result["content"])
    return result


@app.post("/api/admin/tenants")
async def create_tenant(body: TenantCreateRequest, _admin: bool = Depends(require_admin)) -> Dict[str, Any]:
    try:
        risk = ToolRisk(body.allowed_risk)
    except ValueError:
        raise HTTPException(status_code=400, detail="allowed_risk must be safe, guarded or privileged")
    if TENANTS.by_api_key(body.api_key) is not None:
        raise HTTPException(status_code=409, detail="api key already in use")
    tenant = TENANTS.create(body.name, body.api_key, body.token_budget, risk)
    _seed_defaults()
    return {"tenant_id": tenant.tenant_id, "name": tenant.name, "allowed_risk": tenant.allowed_risk.value,
            "token_budget": tenant.token_budget}


@app.get("/api/admin/tenants")
async def list_tenants(_admin: bool = Depends(require_admin)) -> Dict[str, Any]:
    return {"tenants": [{"tenant_id": t.tenant_id, "name": t.name, "token_budget": t.token_budget,
                         "tokens_used": t.tokens_used, "allowed_risk": t.allowed_risk.value}
                        for t in TENANTS.list_all()]}


@app.post("/api/admin/recover")
async def admin_recover(_admin: bool = Depends(require_admin)) -> Dict[str, Any]:
    return {"recovered": SUPERVISOR.recover_orphans(), "active": SUPERVISOR.active()}


@app.get("/api/admin/graph")
async def admin_graph(_admin: bool = Depends(require_admin)) -> Dict[str, Any]:
    return {"graph": GRAPH.describe(), "nodes": [n.value for n in NodeKind]}


@app.get("/", include_in_schema=False)
async def serve_index() -> Response:
    if STATIC_INDEX.exists():
        return FileResponse(str(STATIC_INDEX), media_type="text/html")
    return HTMLResponse(
        "<!doctype html><html><head><meta charset='utf-8'><title>Autonomous Agent Runtime</title></head>"
        "<body style='font-family:system-ui;background:#0b0f14;color:#e6edf3;padding:40px'>"
        "<h1>Autonomous Agent Runtime</h1>"
        "<p>index.html was not found next to main.py. The API is live at "
        "<a style='color:#58a6ff' href='/api/docs'>/api/docs</a>.</p></body></html>",
        status_code=200,
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


if (ROOT_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")


@app.exception_handler(BudgetExceeded)
async def budget_handler(_request: Request, exc: BudgetExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.exception_handler(AuthorizationDenied)
async def authz_handler(_request: Request, exc: AuthorizationDenied) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": str(exc)})


@app.exception_handler(ValidationRejected)
async def validation_handler(_request: Request, exc: ValidationRejected) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.reasons})


def _install_signal_handlers() -> None:
    def handler(signum: int, _frame: Any) -> None:
        LOG.info("received signal %s, draining workers", signum)
        SUPERVISOR.shutdown()
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, AttributeError):
            signal.signal(sig, handler)


def main() -> None:
    import uvicorn

    _install_signal_handlers()
    host = os.environ.get("AGENT_HOST", "0.0.0.0")
    port = int(os.environ.get("AGENT_PORT", "8000"))
    LOG.info("starting Autonomous Agent Runtime on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("LOG_LEVEL", "info").lower(),
                timeout_keep_alive=75, access_log=False)


if __name__ == "__main__":
    main()
