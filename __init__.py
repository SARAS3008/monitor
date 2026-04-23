import copy
import json
import traceback
from datetime import datetime
from pathlib import Path

from aiohttp import web

from .api_routes import register_routes
from .db import Database

WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

PLUGIN_DIR = Path(__file__).resolve().parent


def _resolve_data_dir():
    try:
        import folder_paths  # type: ignore
        user_dir = Path(folder_paths.get_user_directory())
        p = user_dir / "default" / "task_monitor"
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        p = PLUGIN_DIR / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p


DB_PATH = str(_resolve_data_dir() / "task_monitor.db")


class TaskMonitor:
    def __init__(self, db: Database):
        self.db = db
        self.current_prompt_id = None
        self.current_node_id = None
        self.current_node_type = None
        self.queue_remaining = 0
        self.latest_error = None
        self.prompt_cache = {}
        self.node_type_cache = {}
        self.executed_nodes = {}
        self.debug_log = []
        self.last_submission_body = None

    def _log(self, msg: str):
        line = f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        self.debug_log.append(line)
        self.debug_log = self.debug_log[-200:]
        print(f"[TaskMonitor] {msg}")

    def cache_submission(self, prompt_id: str, payload: dict, queue_position=None):
        if prompt_id is None:
            return
        prompt_id = str(prompt_id)
        self.prompt_cache[prompt_id] = copy.deepcopy(payload or {})
        workflow = (payload or {}).get("prompt") or {}
        if isinstance(workflow, dict):
            self.node_type_cache[prompt_id] = {
                str(node_id): (node_data or {}).get("class_type")
                for node_id, node_data in workflow.items()
                if isinstance(node_data, dict)
            }
        repeat_guess = self.db.peek_repeat_count(payload or {})
        self.db.upsert_task_submission(prompt_id, payload or {}, queue_position=queue_position)
        self._log(f"submission cached: prompt_id={prompt_id} queue_position={queue_position} repeat_guess={repeat_guess}")

    def get_node_type(self, prompt_id: str, node_id):
        if prompt_id is None or node_id is None:
            return None
        return (self.node_type_cache.get(str(prompt_id)) or {}).get(str(node_id))

    def _hydrate_from_cache(self, prompt_id: str):
        payload = self.prompt_cache.get(str(prompt_id))
        if payload:
            self.db.hydrate_task_from_payload(str(prompt_id), payload)

    def handle_event(self, event_type: str, payload: dict):
        payload = payload or {}
        prompt_id = payload.get("prompt_id")
        node_id = payload.get("node") if "node" in payload else payload.get("node_id")
        node_type = payload.get("node_type") or self.get_node_type(prompt_id, node_id)
        value = payload.get("value")
        max_value = payload.get("max")

        if event_type == "status":
            queue_remaining = (((payload or {}).get("exec_info") or {}).get("queue_remaining"))
            if queue_remaining is not None:
                self.queue_remaining = queue_remaining
                self.db.update_queue_remaining(queue_remaining)
            return

        if prompt_id is not None:
            prompt_id = str(prompt_id)
            self.db.ensure_task(prompt_id)
            self._hydrate_from_cache(prompt_id)
            self.db.append_event(
                prompt_id=prompt_id,
                event_type=event_type,
                payload=payload,
                node_id=node_id,
                node_type=node_type,
                progress_value=value,
                progress_max=max_value,
            )

        if event_type == "execution_start" and prompt_id is not None:
            prompt_id = str(prompt_id)
            self.current_prompt_id = prompt_id
            self.executed_nodes.setdefault(prompt_id, set())
            run_index = self.db.mark_running(prompt_id)
            self._log(f"execution_start: prompt_id={prompt_id} run_index={run_index}")
            return

        if event_type == "executing" and prompt_id is not None:
            prompt_id = str(prompt_id)
            self.current_prompt_id = prompt_id
            self.current_node_id = str(node_id) if node_id is not None else None
            self.current_node_type = node_type
            self.db.update_executing_node(prompt_id, node_id=node_id, node_type=node_type)
            return

        if event_type == "progress" and prompt_id is not None:
            self.db.update_progress(str(prompt_id), value=value, max_value=max_value, node_id=node_id)
            return

        if event_type in ("executed", "execution_cached") and prompt_id is not None:
            prompt_id = str(prompt_id)
            s = self.executed_nodes.setdefault(prompt_id, set())
            node_key = str(node_id) if node_id is not None else None
            if node_key and node_key not in s:
                s.add(node_key)
                self.db.increment_executed_nodes(prompt_id)
            return

        if event_type == "execution_success" and prompt_id is not None:
            prompt_id = str(prompt_id)
            self.current_prompt_id = None
            self.current_node_id = None
            self.current_node_type = None
            run_index = self.db.mark_success(prompt_id, result_json=payload)
            self._log(f"execution_success: prompt_id={prompt_id} run_index={run_index}")
            return

        if event_type == "execution_error" and prompt_id is not None:
            prompt_id = str(prompt_id)
            error_message = payload.get("exception_message") or payload.get("error") or "execution_error"
            self.latest_error = {
                "prompt_id": prompt_id,
                "message": error_message,
                "payload": payload,
                "at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }
            run_index = self.db.mark_failure(
                prompt_id,
                error_message=error_message,
                error_detail=payload,
                error_node_id=payload.get("node_id") or node_id,
                error_node_type=node_type,
                interrupted=False,
            )
            self.current_prompt_id = None
            self.current_node_id = None
            self.current_node_type = None
            self._log(f"execution_error: prompt_id={prompt_id} run_index={run_index} message={error_message}")
            return

        if event_type == "execution_interrupted" and prompt_id is not None:
            prompt_id = str(prompt_id)
            run_index = self.db.mark_failure(
                prompt_id,
                error_message="execution_interrupted",
                error_detail=payload,
                error_node_id=payload.get("node_id") or node_id,
                error_node_type=node_type,
                interrupted=True,
            )
            self.current_prompt_id = None
            self.current_node_id = None
            self.current_node_type = None
            self._log(f"execution_interrupted: prompt_id={prompt_id} run_index={run_index}")
            return

    def get_status(self):
        return {
            "db_path": DB_PATH,
            "memory": {
                "current_prompt_id": self.current_prompt_id,
                "current_node_id": self.current_node_id,
                "current_node_type": self.current_node_type,
                "queue_remaining": self.queue_remaining,
                "latest_error": self.latest_error,
                "debug_log": self.debug_log[-30:],
                "cached_prompt_count": len(self.prompt_cache),
                "last_submission_body": self.last_submission_body,
            },
            "database": self.db.get_instance_status(),
        }


_db = Database(DB_PATH)
monitor = TaskMonitor(_db)


def _patch_prompt_submission():
    try:
        from server import PromptServer  # type: ignore
    except Exception:
        monitor._log("cannot import PromptServer for prompt hook")
        traceback.print_exc()
        return

    app = PromptServer.instance.app
    if getattr(app, "_task_monitor_prompt_middleware_installed", False):
        return

    @web.middleware
    async def task_monitor_prompt_middleware(request, handler):
        if request.path == "/prompt" and request.method.upper() == "POST":
            try:
                body = await request.json()
                request["task_monitor_payload"] = body
                monitor.last_submission_body = {
                    "keys": list((body or {}).keys())[:50],
                    "repeat_guess": monitor.db.peek_repeat_count(body or {}),
                    "top_level_number": (body or {}).get("number"),
                    "extra_data_keys": list(((body or {}).get("extra_data") or {}).keys())[:30],
                }
                monitor._log(f"captured /prompt request body repeat_guess={monitor.last_submission_body['repeat_guess']}")
            except Exception:
                request["task_monitor_payload"] = None
        response = await handler(request)
        if request.path == "/prompt" and request.method.upper() == "POST":
            try:
                payload = request.get("task_monitor_payload") or {}
                data = None
                if hasattr(response, "text") and response.text:
                    data = json.loads(response.text)
                elif hasattr(response, "body") and response.body:
                    body = response.body.decode("utf-8") if isinstance(response.body, (bytes, bytearray)) else response.body
                    data = json.loads(body)
                if isinstance(data, dict) and data.get("prompt_id") is not None:
                    monitor.cache_submission(data.get("prompt_id"), payload, queue_position=data.get("number"))
                else:
                    monitor._log("/prompt response captured but no prompt_id found")
            except Exception:
                monitor._log("failed to cache prompt submission")
                traceback.print_exc()
        return response

    app.middlewares.append(task_monitor_prompt_middleware)
    app._task_monitor_prompt_middleware_installed = True
    monitor._log("prompt middleware installed")


def _patch_send_sync():
    try:
        from server import PromptServer  # type: ignore
    except Exception:
        monitor._log("cannot import PromptServer")
        traceback.print_exc()
        return

    ps = PromptServer.instance
    if getattr(ps, "_task_monitor_send_sync_patched", False):
        return

    original_send_sync = ps.send_sync

    def patched_send_sync(event, data, *args, **kwargs):
        try:
            if event in {
                "status",
                "execution_start",
                "executing",
                "progress",
                "executed",
                "execution_cached",
                "execution_success",
                "execution_error",
                "execution_interrupted",
            }:
                monitor.handle_event(event, data)
        except Exception:
            monitor._log(f"error handling event: {event}")
            traceback.print_exc()
        return original_send_sync(event, data, *args, **kwargs)

    ps.send_sync = patched_send_sync
    ps._task_monitor_send_sync_patched = True
    monitor._log("send_sync patched")


def _patch_progress_hook():
    try:
        import comfy.utils  # type: ignore
    except Exception:
        monitor._log("comfy.utils not available; skip progress hook")
        return

    if getattr(comfy.utils, "_task_monitor_progress_hook_installed", False):
        return

    def hook(value, max_value, preview=None):
        try:
            prompt_id = monitor.current_prompt_id
            node_id = monitor.current_node_id
            if prompt_id is not None:
                monitor.handle_event(
                    "progress",
                    {
                        "prompt_id": prompt_id,
                        "node": node_id,
                        "value": value,
                        "max": max_value,
                    },
                )
        except Exception:
            traceback.print_exc()

    try:
        comfy.utils.set_progress_bar_global_hook(hook)
        comfy.utils._task_monitor_progress_hook_installed = True
        monitor._log("progress hook installed")
    except Exception:
        monitor._log("failed to install progress hook")
        traceback.print_exc()


def _register_static_routes():
    try:
        from server import PromptServer  # type: ignore
    except Exception:
        monitor._log("cannot import PromptServer for routes")
        traceback.print_exc()
        return

    prompt_server = PromptServer.instance
    register_routes(prompt_server, monitor)

    static_path = str((PLUGIN_DIR / "web").resolve())
    prompt_server.routes.static("/task_monitor", static_path)
    monitor._log(f"static route registered: /task_monitor -> {static_path}")


try:
    _patch_prompt_submission()
    _patch_send_sync()
    _patch_progress_hook()
    _register_static_routes()
    monitor._log(f"initialized. DB: {DB_PATH}")
except Exception:
    monitor._log("initialization failed")
    traceback.print_exc()
