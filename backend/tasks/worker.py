import os, time, requests
from celery import Celery; from dotenv import load_dotenv; from app import crud
load_dotenv()
celery_app = Celery("tasks", broker=os.getenv("CELERY_BROKER_URL"), backend=os.getenv("CELERY_RESULT_BACKEND"))
@celery_app.task(name="create_ai_task", bind=True, max_retries=2)
def create_ai_task(self, task_id: int, prompt: str, worker_url: str):
    crud.update_task_status(task_id, "RUNNING")
    try:
        res_a = requests.post(f"{worker_url}/execute", json={"prompt": prompt}, timeout=30); res_a.raise_for_status()
        crud.update_task_status(task_id, "VALIDATING", r_a=res_a.json()["result"])
        res_b = requests.post(f"{worker_url}/validate", json={"prompt": prompt, "original_result": res_a.json()["result"]}, timeout=30); res_b.raise_for_status()
        crud.update_task_status(task_id, "COMPLETED", r_a=res_a.json()["result"], r_b=res_b.json()["result"])
        return {"status": "SUCCESS"}
    except requests.exceptions.RequestException as exc:
        crud.update_task_status(task_id, "FAILED"); raise self.retry(exc=exc)
