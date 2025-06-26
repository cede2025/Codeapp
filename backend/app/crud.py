from . import schemas
tasks_db, task_id_counter = {}, 1
def create_task(task: schemas.TaskCreate) -> int:
    global task_id_counter; task_id = task_id_counter
    tasks_db[task_id] = {"id": task_id, "prompt": task.prompt, "status": "QUEUED", "result_a": None, "result_b": None}
    task_id_counter += 1; return task_id
def update_task_status(task_id: int, status: str, r_a: str=None, r_b: str=None):
    if task_id in tasks_db:
        tasks_db[task_id]['status'] = status
        if r_a: tasks_db[task_id]['result_a'] = r_a
        if r_b: tasks_db[task_id]['result_b'] = r_b
