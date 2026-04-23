import json

from aiohttp import web


def _json_response(data, status=200):
    return web.Response(text=json.dumps(data, ensure_ascii=False, default=str), status=status, content_type="application/json")


def register_routes(prompt_server, monitor):
    routes = prompt_server.routes

    @routes.get("/task_monitor/status")
    async def task_monitor_status(request):
        return _json_response(monitor.get_status())

    @routes.get("/task_monitor/tasks")
    async def task_monitor_tasks(request):
        q = request.rel_url.query
        data = monitor.db.list_tasks(
            status=q.get("status"),
            workflow_name=q.get("workflow_name"),
            model_name=q.get("model_name"),
            task_type=q.get("task_type"),
            limit=min(int(q.get("limit", 50)), 200),
            offset=max(int(q.get("offset", 0)), 0),
        )
        return _json_response(data)

    @routes.get(r"/task_monitor/tasks/{prompt_id}")
    async def task_monitor_task_detail(request):
        prompt_id = request.match_info.get("prompt_id")
        data = monitor.db.get_task_detail(prompt_id)
        if not data:
            return _json_response({"error": "task not found", "prompt_id": prompt_id}, status=404)
        return _json_response(data)

    @routes.post(r"/task_monitor/tasks/{prompt_id}/repeat_count")
    async def task_monitor_set_repeat(request):
        prompt_id = request.match_info.get("prompt_id")
        body = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        count = body.get("repeat_count") or request.rel_url.query.get("repeat_count")
        try:
            count = max(int(count), 1)
        except Exception:
            return _json_response({"error": "repeat_count must be integer >= 1"}, status=400)
        monitor.db.set_repeat_count(prompt_id, count)
        return _json_response({"ok": True, "prompt_id": prompt_id, "repeat_count": count})

    @routes.get("/task_monitor/errors")
    async def task_monitor_errors(request):
        q = request.rel_url.query
        limit = min(int(q.get("limit", 50)), 200)
        return _json_response({"items": monitor.db.get_error_groups(limit=limit)})

    @routes.get("/task_monitor/stats/overview")
    async def task_monitor_overview(request):
        return _json_response(monitor.db.get_overview_stats())

    @routes.get("/task_monitor/stats/workflows")
    async def task_monitor_workflows(request):
        q = request.rel_url.query
        limit = min(int(q.get("limit", 50)), 200)
        return _json_response({"items": monitor.db.get_workflow_stats(limit=limit)})

    @routes.get("/task_monitor/debug")
    async def task_monitor_debug(request):
        return _json_response({
            "db_path": monitor.db.db_path,
            "memory": monitor.get_status().get("memory", {}),
            "database": monitor.get_status().get("database", {}),
        })
