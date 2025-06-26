import os, asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette_prometheus import metrics, PrometheusMiddleware
from app import models, schemas, crud
from app.database import engine
from tasks.worker import create_ai_task
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="AI Command Center - Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(PrometheusMiddleware); app.add_route("/metrics", metrics)
class ConnectionManager:
    def __init__(self): self.active_connections: list[WebSocket] = []
    async def connect(self, ws: WebSocket): await ws.accept(); self.active_connections.append(ws)
    def disconnect(self, ws: WebSocket): self.active_connections.remove(ws)
    async def broadcast(self, msg: str):
        for conn in self.active_connections: await conn.send_text(msg)
manager = ConnectionManager()
@app.post("/tasks/", response_model=schemas.Task, status_code=201)
def create_task_endpoint(task_in: schemas.TaskCreate):
    task_id = crud.create_task(task_in)
    create_ai_task.delay(task_id, task_in.prompt, os.getenv("WORKER_GPT_URL"))
    asyncio.run(manager.broadcast(f"New task #{task_id} created."))
    return {"id": task_id, **task_in.dict(), "status": "QUEUED"}
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: manager.disconnect(ws)
