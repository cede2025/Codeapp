import os, time, random; from fastapi import FastAPI; from pydantic import BaseModel; from starlette_prometheus import metrics, PrometheusMiddleware
app = FastAPI(); app.add_middleware(PrometheusMiddleware); app.add_route("/metrics", metrics)
MODEL_NAME = os.getenv("MODEL_NAME", "Default")
class Req(BaseModel): prompt: str
class ValReq(Req): original_result: str
@app.post("/execute")
def execute(r:Req): time.sleep(random.uniform(1,2)); return {"result":f"[{MODEL_NAME}] Result for: {r.prompt[:30]}..."}
@app.post("/validate")
def validate(r:ValReq): time.sleep(random.uniform(0,1)); return {"result":f"[{MODEL_NAME}] Validation for: {r.original_result[:30]}..."}
