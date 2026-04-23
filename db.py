import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime


def utcnow_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _table_columns(self, conn, table_name: str):
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row["name"] for row in rows}

    def _add_column_if_missing(self, conn, table_name: str, column_name: str, ddl: str):
        cols = self._table_columns(conn, table_name)
        if column_name not in cols:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")

    def _init_db(self):
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tm_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id TEXT UNIQUE,
                    client_id TEXT,
                    user_id TEXT,
                    app_id TEXT,
                    workflow_id TEXT,
                    workflow_name TEXT,
                    workflow_hash TEXT,
                    model_name TEXT,
                    task_type TEXT,
                    status TEXT,
                    queue_position INTEGER,
                    queue_remaining INTEGER,
                    total_nodes INTEGER,
                    executed_nodes INTEGER DEFAULT 0,
                    current_node_id TEXT,
                    current_node_type TEXT,
                    progress_value INTEGER,
                    progress_max INTEGER,
                    progress_percent REAL,
                    repeat_count INTEGER DEFAULT 1,
                    batch_size INTEGER,
                    current_run_index INTEGER,
                    completed_runs INTEGER DEFAULT 0,
                    failed_runs INTEGER DEFAULT 0,
                    input_payload_json TEXT,
                    workflow_json TEXT,
                    result_json TEXT,
                    error_message TEXT,
                    error_detail_json TEXT,
                    error_node_id TEXT,
                    error_node_type TEXT,
                    submitted_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    duration_ms INTEGER,
                    updated_at TEXT
                )
                """
            )
            for col, ddl in [
                ("repeat_count", "repeat_count INTEGER DEFAULT 1"),
                ("batch_size", "batch_size INTEGER"),
                ("workflow_hash", "workflow_hash TEXT"),
                ("task_type", "task_type TEXT"),
                ("current_run_index", "current_run_index INTEGER"),
                ("completed_runs", "completed_runs INTEGER DEFAULT 0"),
                ("failed_runs", "failed_runs INTEGER DEFAULT 0"),
            ]:
                self._add_column_if_missing(conn, "tm_tasks", col, ddl)

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tm_task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id TEXT,
                    event_type TEXT,
                    node_id TEXT,
                    node_type TEXT,
                    progress_value INTEGER,
                    progress_max INTEGER,
                    payload_json TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tm_task_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_id TEXT NOT NULL,
                    run_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    current_node_id TEXT,
                    current_node_type TEXT,
                    progress_value INTEGER,
                    progress_max INTEGER,
                    progress_percent REAL,
                    error_message TEXT,
                    error_detail_json TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    duration_ms INTEGER,
                    updated_at TEXT,
                    UNIQUE(prompt_id, run_index)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tm_instance_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    queue_remaining INTEGER DEFAULT 0,
                    current_prompt_id TEXT,
                    current_node_id TEXT,
                    current_node_type TEXT,
                    latest_error_message TEXT,
                    latest_error_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute("INSERT OR IGNORE INTO tm_instance_state (id, updated_at) VALUES (1, ?)", (utcnow_str(),))
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_tasks_status ON tm_tasks(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_tasks_workflow_name ON tm_tasks(workflow_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_tasks_model_name ON tm_tasks(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_task_events_prompt_id ON tm_task_events(prompt_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_runs_prompt ON tm_task_runs(prompt_id)")

    def _safe_json(self, obj):
        return json.dumps(obj, ensure_ascii=False) if obj is not None else None

    def _parse_int(self, v):
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return None

    def _search_repeat_candidates(self, obj, path="root", out=None):
        if out is None:
            out = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                p = f"{path}.{k}"
                iv = self._parse_int(v)
                if iv is not None and iv > 1 and any(tok in lk for tok in ["repeat", "count", "times", "runs", "queue", "batch_count"]):
                    out.append((p, iv))
                self._search_repeat_candidates(v, p, out)
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                self._search_repeat_candidates(v, f"{path}[{idx}]", out)
        return out

    def peek_repeat_count(self, payload: dict):
        meta = self._extract_meta(payload)
        return meta.get("repeat_count")

    def _extract_meta(self, payload: dict):
        payload = payload or {}
        extra = payload.get("extra_data") or {}
        task_meta = extra.get("task_monitor") or {}
        workflow = payload.get("prompt") or {}

        workflow_name = task_meta.get("workflow_name") or extra.get("workflow_name") or payload.get("workflow_name")
        workflow_id = task_meta.get("workflow_id") or payload.get("workflow_id")
        task_type = task_meta.get("task_type") or payload.get("task_type")
        user_id = task_meta.get("user_id") or payload.get("user_id")
        app_id = task_meta.get("app_id") or payload.get("app_id")
        model_name = task_meta.get("model_name") or payload.get("model_name")
        batch_size = task_meta.get("batch_size")
        repeat_count = task_meta.get("repeat_count") or payload.get("repeat_count") or payload.get("runs") or payload.get("queue_count")

        if not isinstance(workflow, dict):
            workflow = {}

        inferred_batch_sizes = []
        inferred_models = []
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type") or ""
            inputs = node.get("inputs") or {}
            if not workflow_name and class_type in {"SaveImage", "SaveAnimatedWEBP", "VHS_VideoCombine"}:
                prefix = inputs.get("filename_prefix") or inputs.get("file_prefix")
                if prefix:
                    workflow_name = str(prefix)
            if not model_name and "CheckpointLoader" in class_type:
                ckpt = inputs.get("ckpt_name") or inputs.get("checkpoint")
                if ckpt:
                    inferred_models.append(str(ckpt))
            if not task_type:
                if class_type in {"VHS_VideoCombine", "SaveAnimatedWEBP"}:
                    task_type = "video"
                elif class_type == "SaveImage":
                    task_type = task_type or "image"
            if batch_size is None:
                for key in ("batch_size", "length", "video_frames", "frame_count"):
                    if isinstance(inputs.get(key), int):
                        inferred_batch_sizes.append(int(inputs.get(key)))

        if not model_name and inferred_models:
            model_name = inferred_models[0]
        if batch_size is None and inferred_batch_sizes:
            batch_size = max(inferred_batch_sizes)

        candidates = self._search_repeat_candidates(payload)
        candidate_values = [v for _, v in candidates if v > 1 and v <= 256]
        if repeat_count is None and candidate_values:
            repeat_count = max(candidate_values)

        workflow_hash = None
        try:
            workflow_hash = hashlib.sha1(json.dumps(workflow, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        except Exception:
            workflow_hash = None

        total_nodes = len(workflow) if isinstance(workflow, dict) else None
        return {
            "client_id": str(payload.get("client_id") or "") or None,
            "user_id": user_id,
            "app_id": app_id,
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "workflow_hash": workflow_hash,
            "model_name": model_name,
            "task_type": task_type,
            "repeat_count": int(repeat_count) if isinstance(self._parse_int(repeat_count), int) and self._parse_int(repeat_count) > 0 else 1,
            "batch_size": int(batch_size) if isinstance(self._parse_int(batch_size), int) and self._parse_int(batch_size) > 0 else None,
            "total_nodes": total_nodes,
            "workflow": workflow,
            "repeat_candidates": candidates[:50],
        }

    def ensure_task(self, prompt_id: str, status: str = "queued"):
        now = utcnow_str()
        with self._lock, self.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO tm_tasks (prompt_id, status, submitted_at, updated_at, repeat_count) VALUES (?, ?, ?, ?, 1)",
                (str(prompt_id), status, now, now),
            )
            self._ensure_runs_for_task(conn, str(prompt_id), 1)

    def _ensure_runs_for_task(self, conn, prompt_id: str, repeat_count: int):
        repeat_count = max(int(repeat_count or 1), 1)
        existing = conn.execute("SELECT COUNT(1) AS cnt FROM tm_task_runs WHERE prompt_id=?", (prompt_id,)).fetchone()["cnt"]
        now = utcnow_str()
        for i in range(existing + 1, repeat_count + 1):
            conn.execute(
                "INSERT OR IGNORE INTO tm_task_runs (prompt_id, run_index, status, updated_at) VALUES (?, ?, 'queued', ?)",
                (prompt_id, i, now),
            )

    def upsert_task_submission(self, prompt_id: str, payload: dict, queue_position=None):
        meta = self._extract_meta(payload)
        now = utcnow_str()
        repeat_count = meta.get("repeat_count") or 1
        with self._lock, self.connection() as conn:
            conn.execute(
                """
                INSERT INTO tm_tasks (
                    prompt_id, client_id, user_id, app_id, workflow_id, workflow_name,
                    workflow_hash, model_name, task_type, status, queue_position,
                    total_nodes, repeat_count, batch_size, input_payload_json, workflow_json,
                    submitted_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prompt_id) DO UPDATE SET
                    client_id=COALESCE(excluded.client_id, tm_tasks.client_id),
                    user_id=COALESCE(excluded.user_id, tm_tasks.user_id),
                    app_id=COALESCE(excluded.app_id, tm_tasks.app_id),
                    workflow_id=COALESCE(excluded.workflow_id, tm_tasks.workflow_id),
                    workflow_name=COALESCE(excluded.workflow_name, tm_tasks.workflow_name),
                    workflow_hash=COALESCE(excluded.workflow_hash, tm_tasks.workflow_hash),
                    model_name=COALESCE(excluded.model_name, tm_tasks.model_name),
                    task_type=COALESCE(excluded.task_type, tm_tasks.task_type),
                    status=CASE WHEN tm_tasks.status IN ('success','failed','interrupted','running') THEN tm_tasks.status ELSE 'queued' END,
                    queue_position=COALESCE(excluded.queue_position, tm_tasks.queue_position),
                    total_nodes=COALESCE(excluded.total_nodes, tm_tasks.total_nodes),
                    repeat_count=CASE WHEN COALESCE(excluded.repeat_count, 1) > COALESCE(tm_tasks.repeat_count, 1) THEN excluded.repeat_count ELSE tm_tasks.repeat_count END,
                    batch_size=COALESCE(excluded.batch_size, tm_tasks.batch_size),
                    input_payload_json=COALESCE(excluded.input_payload_json, tm_tasks.input_payload_json),
                    workflow_json=COALESCE(excluded.workflow_json, tm_tasks.workflow_json),
                    updated_at=excluded.updated_at
                """,
                (
                    str(prompt_id), meta.get("client_id"), meta.get("user_id"), meta.get("app_id"), meta.get("workflow_id"), meta.get("workflow_name"),
                    meta.get("workflow_hash"), meta.get("model_name"), meta.get("task_type"), "queued", queue_position,
                    meta.get("total_nodes"), repeat_count, meta.get("batch_size"), self._safe_json(payload), self._safe_json(meta.get("workflow")),
                    now, now,
                ),
            )
            self._ensure_runs_for_task(conn, str(prompt_id), repeat_count)

    def hydrate_task_from_payload(self, prompt_id: str, payload: dict):
        meta = self._extract_meta(payload)
        with self._lock, self.connection() as conn:
            conn.execute(
                """
                UPDATE tm_tasks SET
                    client_id=COALESCE(?, client_id),
                    user_id=COALESCE(?, user_id),
                    app_id=COALESCE(?, app_id),
                    workflow_id=COALESCE(?, workflow_id),
                    workflow_name=COALESCE(?, workflow_name),
                    workflow_hash=COALESCE(?, workflow_hash),
                    model_name=COALESCE(?, model_name),
                    task_type=COALESCE(?, task_type),
                    total_nodes=COALESCE(?, total_nodes),
                    repeat_count=CASE WHEN COALESCE(?, 1) > COALESCE(repeat_count, 1) THEN ? ELSE COALESCE(repeat_count, 1) END,
                    batch_size=COALESCE(?, batch_size),
                    input_payload_json=COALESCE(?, input_payload_json),
                    workflow_json=COALESCE(?, workflow_json),
                    updated_at=?
                WHERE prompt_id=?
                """,
                (
                    meta.get("client_id"), meta.get("user_id"), meta.get("app_id"), meta.get("workflow_id"), meta.get("workflow_name"),
                    meta.get("workflow_hash"), meta.get("model_name"), meta.get("task_type"), meta.get("total_nodes"),
                    meta.get("repeat_count"), meta.get("repeat_count"), meta.get("batch_size"), self._safe_json(payload), self._safe_json(meta.get("workflow")), utcnow_str(), str(prompt_id),
                ),
            )
            self._ensure_runs_for_task(conn, str(prompt_id), meta.get("repeat_count") or 1)

    def set_repeat_count(self, prompt_id: str, repeat_count: int):
        repeat_count = max(int(repeat_count), 1)
        with self._lock, self.connection() as conn:
            self.ensure_task(prompt_id)
            conn.execute("UPDATE tm_tasks SET repeat_count=?, updated_at=? WHERE prompt_id=?", (repeat_count, utcnow_str(), str(prompt_id)))
            self._ensure_runs_for_task(conn, str(prompt_id), repeat_count)

    def append_event(self, prompt_id: str, event_type: str, payload: dict, node_id=None, node_type=None, progress_value=None, progress_max=None):
        with self._lock, self.connection() as conn:
            conn.execute(
                "INSERT INTO tm_task_events (prompt_id, event_type, node_id, node_type, progress_value, progress_max, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(prompt_id), event_type, str(node_id) if node_id is not None else None, node_type, progress_value, progress_max, self._safe_json(payload), utcnow_str()),
            )

    def _get_active_run(self, conn, prompt_id: str):
        row = conn.execute("SELECT * FROM tm_task_runs WHERE prompt_id=? AND status='running' ORDER BY run_index LIMIT 1", (str(prompt_id),)).fetchone()
        return row

    def _next_runnable_run(self, conn, prompt_id: str):
        row = conn.execute("SELECT * FROM tm_task_runs WHERE prompt_id=? AND status IN ('queued','submitted') ORDER BY run_index LIMIT 1", (str(prompt_id),)).fetchone()
        if row:
            return row
        return conn.execute("SELECT * FROM tm_task_runs WHERE prompt_id=? ORDER BY run_index DESC LIMIT 1", (str(prompt_id),)).fetchone()

    def mark_running(self, prompt_id: str):
        self.ensure_task(prompt_id)
        now = utcnow_str()
        with self._lock, self.connection() as conn:
            row = self._get_active_run(conn, prompt_id)
            if not row:
                task = conn.execute("SELECT repeat_count FROM tm_tasks WHERE prompt_id=?", (str(prompt_id),)).fetchone()
                repeat_count = (task["repeat_count"] if task else 1) or 1
                self._ensure_runs_for_task(conn, str(prompt_id), repeat_count)
                row = self._next_runnable_run(conn, prompt_id)
                if row:
                    conn.execute(
                        "UPDATE tm_task_runs SET status='running', started_at=COALESCE(started_at, ?), updated_at=? WHERE prompt_id=? AND run_index=?",
                        (now, now, str(prompt_id), row["run_index"]),
                    )
            run_index = row["run_index"] if row else 1
            conn.execute(
                """
                UPDATE tm_tasks
                SET status='running', started_at=COALESCE(started_at, ?), current_run_index=?, updated_at=?
                WHERE prompt_id=?
                """,
                (now, run_index, now, str(prompt_id)),
            )
            conn.execute("UPDATE tm_instance_state SET current_prompt_id=?, updated_at=? WHERE id=1", (str(prompt_id), now))
            return run_index

    def update_executing_node(self, prompt_id: str, node_id=None, node_type=None):
        self.ensure_task(prompt_id)
        now = utcnow_str()
        with self._lock, self.connection() as conn:
            conn.execute(
                "UPDATE tm_tasks SET current_node_id=?, current_node_type=?, updated_at=? WHERE prompt_id=?",
                (str(node_id) if node_id is not None else None, node_type, now, str(prompt_id)),
            )
            active = self._get_active_run(conn, prompt_id)
            if active:
                conn.execute(
                    "UPDATE tm_task_runs SET current_node_id=?, current_node_type=?, updated_at=? WHERE prompt_id=? AND run_index=?",
                    (str(node_id) if node_id is not None else None, node_type, now, str(prompt_id), active["run_index"]),
                )
            conn.execute(
                "UPDATE tm_instance_state SET current_prompt_id=?, current_node_id=?, current_node_type=?, updated_at=? WHERE id=1",
                (str(prompt_id), str(node_id) if node_id is not None else None, node_type, now),
            )

    def update_progress(self, prompt_id: str, value=None, max_value=None, node_id=None):
        self.ensure_task(prompt_id)
        percent = round((float(value) * 100.0 / float(max_value)), 2) if isinstance(value, int) and isinstance(max_value, int) and max_value > 0 else None
        now = utcnow_str()
        with self._lock, self.connection() as conn:
            conn.execute(
                "UPDATE tm_tasks SET progress_value=?, progress_max=?, progress_percent=?, current_node_id=COALESCE(?, current_node_id), updated_at=? WHERE prompt_id=?",
                (value, max_value, percent, str(node_id) if node_id is not None else None, now, str(prompt_id)),
            )
            active = self._get_active_run(conn, prompt_id)
            if active:
                conn.execute(
                    "UPDATE tm_task_runs SET progress_value=?, progress_max=?, progress_percent=?, current_node_id=COALESCE(?, current_node_id), updated_at=? WHERE prompt_id=? AND run_index=?",
                    (value, max_value, percent, str(node_id) if node_id is not None else None, now, str(prompt_id), active["run_index"]),
                )

    def increment_executed_nodes(self, prompt_id: str):
        self.ensure_task(prompt_id)
        with self._lock, self.connection() as conn:
            conn.execute(
                """
                UPDATE tm_tasks
                SET executed_nodes = COALESCE(executed_nodes, 0) + 1,
                    progress_value = CASE WHEN COALESCE(progress_value, 0) < COALESCE(executed_nodes, 0) + 1 THEN COALESCE(executed_nodes, 0) + 1 ELSE progress_value END,
                    progress_max = COALESCE(progress_max, total_nodes),
                    progress_percent = CASE WHEN COALESCE(total_nodes, 0) > 0 THEN ROUND(((COALESCE(executed_nodes, 0) + 1) * 100.0) / total_nodes, 2) ELSE progress_percent END,
                    updated_at=?
                WHERE prompt_id=?
                """,
                (utcnow_str(), str(prompt_id)),
            )

    def _compute_duration_ms(self, started_at):
        if not started_at:
            return None
        try:
            started = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
            return int((datetime.utcnow() - started).total_seconds() * 1000)
        except Exception:
            return None

    def mark_success(self, prompt_id: str, result_json=None):
        self.ensure_task(prompt_id)
        now = utcnow_str()
        run_index = 1
        with self._lock, self.connection() as conn:
            active = self._get_active_run(conn, prompt_id)
            if not active:
                active = self._next_runnable_run(conn, prompt_id)
                if active:
                    conn.execute("UPDATE tm_task_runs SET status='running', started_at=COALESCE(started_at, ?), updated_at=? WHERE prompt_id=? AND run_index=?", (now, now, str(prompt_id), active["run_index"]))
            if active:
                run_index = active["run_index"]
                duration_ms = self._compute_duration_ms(active["started_at"] or now)
                conn.execute(
                    "UPDATE tm_task_runs SET status='success', finished_at=?, duration_ms=?, updated_at=? WHERE prompt_id=? AND run_index=?",
                    (now, duration_ms, now, str(prompt_id), run_index),
                )
            row = conn.execute("SELECT started_at, repeat_count FROM tm_tasks WHERE prompt_id=?", (str(prompt_id),)).fetchone()
            duration_ms = self._compute_duration_ms(row["started_at"] if row else None)
            counts = conn.execute(
                "SELECT SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS success_runs, SUM(CASE WHEN status IN ('failed','interrupted') THEN 1 ELSE 0 END) AS failed_runs FROM tm_task_runs WHERE prompt_id=?",
                (str(prompt_id),),
            ).fetchone()
            success_runs = counts["success_runs"] or 0
            failed_runs = counts["failed_runs"] or 0
            repeat_count = (row["repeat_count"] if row else 1) or 1
            parent_status = 'success' if (success_runs + failed_runs) >= repeat_count and failed_runs == 0 else ('running' if (success_runs + failed_runs) < repeat_count else 'failed')
            conn.execute(
                "UPDATE tm_tasks SET status=?, completed_runs=?, failed_runs=?, current_run_index=NULL, result_json=?, finished_at=CASE WHEN ? IN ('success','failed','interrupted') THEN ? ELSE finished_at END, duration_ms=?, updated_at=? WHERE prompt_id=?",
                (parent_status, success_runs, failed_runs, self._safe_json(result_json), parent_status, now, duration_ms, now, str(prompt_id)),
            )
            conn.execute("UPDATE tm_instance_state SET current_prompt_id=NULL, current_node_id=NULL, current_node_type=NULL, updated_at=? WHERE id=1", (now,))
        return run_index

    def mark_failure(self, prompt_id: str, error_message=None, error_detail=None, error_node_id=None, error_node_type=None, interrupted=False):
        self.ensure_task(prompt_id)
        now = utcnow_str()
        run_index = 1
        new_status = 'interrupted' if interrupted else 'failed'
        with self._lock, self.connection() as conn:
            active = self._get_active_run(conn, prompt_id)
            if not active:
                active = self._next_runnable_run(conn, prompt_id)
                if active:
                    conn.execute("UPDATE tm_task_runs SET status='running', started_at=COALESCE(started_at, ?), updated_at=? WHERE prompt_id=? AND run_index=?", (now, now, str(prompt_id), active["run_index"]))
            if active:
                run_index = active["run_index"]
                duration_ms = self._compute_duration_ms(active["started_at"] or now)
                conn.execute(
                    "UPDATE tm_task_runs SET status=?, error_message=?, error_detail_json=?, finished_at=?, duration_ms=?, updated_at=? WHERE prompt_id=? AND run_index=?",
                    (new_status, error_message, self._safe_json(error_detail), now, duration_ms, now, str(prompt_id), run_index),
                )
            row = conn.execute("SELECT started_at, repeat_count FROM tm_tasks WHERE prompt_id=?", (str(prompt_id),)).fetchone()
            duration_ms = self._compute_duration_ms(row["started_at"] if row else None)
            counts = conn.execute(
                "SELECT SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS success_runs, SUM(CASE WHEN status IN ('failed','interrupted') THEN 1 ELSE 0 END) AS failed_runs FROM tm_task_runs WHERE prompt_id=?",
                (str(prompt_id),),
            ).fetchone()
            success_runs = counts["success_runs"] or 0
            failed_runs = counts["failed_runs"] or 0
            repeat_count = (row["repeat_count"] if row else 1) or 1
            done = success_runs + failed_runs
            parent_status = new_status if done >= repeat_count else 'running'
            conn.execute(
                "UPDATE tm_tasks SET status=?, completed_runs=?, failed_runs=?, current_run_index=NULL, error_message=?, error_detail_json=?, error_node_id=?, error_node_type=?, finished_at=CASE WHEN ? IN ('failed','interrupted') THEN ? ELSE finished_at END, duration_ms=?, updated_at=? WHERE prompt_id=?",
                (parent_status, success_runs, failed_runs, error_message, self._safe_json(error_detail), str(error_node_id) if error_node_id is not None else None, error_node_type, parent_status, now, duration_ms, now, str(prompt_id)),
            )
            conn.execute(
                "UPDATE tm_instance_state SET current_prompt_id=NULL, current_node_id=NULL, current_node_type=NULL, latest_error_message=?, latest_error_at=?, updated_at=? WHERE id=1",
                (error_message, now, now),
            )
        return run_index

    def update_queue_remaining(self, queue_remaining: int):
        with self._lock, self.connection() as conn:
            now = utcnow_str()
            conn.execute("UPDATE tm_instance_state SET queue_remaining=?, updated_at=? WHERE id=1", (queue_remaining, now))
            conn.execute("UPDATE tm_tasks SET queue_remaining=?, updated_at=? WHERE status IN ('queued', 'running')", (queue_remaining, now))

    def get_instance_status(self):
        with self.connection() as conn:
            state = conn.execute("SELECT * FROM tm_instance_state WHERE id=1").fetchone()
            current = None
            if state and state["current_prompt_id"]:
                current = conn.execute("SELECT * FROM tm_tasks WHERE prompt_id=?", (state["current_prompt_id"],)).fetchone()
            return {"instance": dict(state) if state else {}, "current_task": dict(current) if current else None}

    def list_tasks(self, status=None, workflow_name=None, model_name=None, task_type=None, limit=50, offset=0):
        query = "SELECT * FROM tm_tasks WHERE 1=1"
        params = []
        if status:
            query += " AND status=?"
            params.append(status)
        if workflow_name:
            query += " AND workflow_name LIKE ?"
            params.append(f"%{workflow_name}%")
        if model_name:
            query += " AND model_name LIKE ?"
            params.append(f"%{model_name}%")
        if task_type:
            query += " AND task_type=?"
            params.append(task_type)
        count_query = "SELECT COUNT(1) AS cnt FROM (" + query + ")"
        query += " ORDER BY COALESCE(submitted_at, updated_at) DESC LIMIT ? OFFSET ?"
        params_q = params + [limit, offset]
        with self.connection() as conn:
            rows = conn.execute(query, params_q).fetchall()
            total = conn.execute(count_query, params).fetchone()["cnt"]
            items = []
            for r in rows:
                item = dict(r)
                item["remaining_runs"] = max((item.get("repeat_count") or 1) - (item.get("completed_runs") or 0) - (item.get("failed_runs") or 0), 0)
                items.append(item)
            return {"items": items, "total": total, "limit": limit, "offset": offset}

    def get_task_detail(self, prompt_id: str):
        with self.connection() as conn:
            task = conn.execute("SELECT * FROM tm_tasks WHERE prompt_id=?", (str(prompt_id),)).fetchone()
            if not task:
                return None
            events = conn.execute("SELECT * FROM tm_task_events WHERE prompt_id=? ORDER BY created_at ASC, id ASC", (str(prompt_id),)).fetchall()
            runs = conn.execute("SELECT * FROM tm_task_runs WHERE prompt_id=? ORDER BY run_index ASC", (str(prompt_id),)).fetchall()
            return {"task": dict(task), "events": [dict(r) for r in events], "runs": [dict(r) for r in runs]}

    def get_error_groups(self, limit=50):
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(error_message, 'Unknown Error') AS error_message,
                       COALESCE(error_node_type, 'Unknown Node') AS error_node_type,
                       COALESCE(workflow_name, 'Unknown Workflow') AS workflow_name,
                       COUNT(1) AS count,
                       MAX(finished_at) AS last_seen_at
                FROM tm_tasks
                WHERE status IN ('failed', 'interrupted')
                GROUP BY error_message, error_node_type, workflow_name
                ORDER BY count DESC, last_seen_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_overview_stats(self):
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(1) AS total_count,
                       SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS success_count,
                       SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running_count,
                       SUM(CASE WHEN status='queued' THEN 1 ELSE 0 END) AS queued_count,
                       SUM(COALESCE(repeat_count, 1)) AS total_requested_runs,
                       SUM(COALESCE(completed_runs, 0)) AS completed_runs,
                       SUM(COALESCE(failed_runs, 0)) AS failed_runs
                FROM tm_tasks
                WHERE submitted_at >= datetime('now', '-1 day')
                """
            ).fetchone()
            return dict(row) if row else {}

    def get_workflow_stats(self, limit=50):
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(workflow_name, '-') AS workflow_name,
                       COALESCE(model_name, '-') AS model_name,
                       COALESCE(task_type, '-') AS task_type,
                       COUNT(1) AS request_count,
                       SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS success_count,
                       SUM(CASE WHEN status IN ('failed','interrupted') THEN 1 ELSE 0 END) AS failed_count,
                       ROUND(AVG(duration_ms), 2) AS avg_duration_ms,
                       SUM(COALESCE(repeat_count, 1)) AS total_requested_runs
                FROM tm_tasks
                GROUP BY workflow_name, model_name, task_type
                ORDER BY request_count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
