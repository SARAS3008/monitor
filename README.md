# ComfyUI-TaskMonitor-Enhanced-v3

增强内容：
- 任务持久化
- 事件时间线
- 子执行拆分（tm_task_runs）
- repeat_count 自动识别
- repeat_count 手动修正接口
- 错误聚合、总览统计

接口：
- `/task_monitor/index.html`
- `/task_monitor/tasks`
- `/task_monitor/tasks/{prompt_id}`
- `POST /task_monitor/tasks/{prompt_id}/repeat_count`
- `/task_monitor/errors`
- `/task_monitor/stats/overview`
- `/task_monitor/debug`

说明：
- 这版会尽量从 `/prompt` 请求体里自动猜测批量运行次数。
- 如果你的 ComfyUI 前端版本没有把“右上角运行次数”直接带到请求里，可以在详情页手动设置 repeat_count。
- 设置后会生成对应数量的子执行占位，并在后续 execution_start / success / failure 里依次推进。
