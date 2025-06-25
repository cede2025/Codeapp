from pydantic import BaseModel; from typing import Optional
class TaskBase(BaseModel): prompt: str
class TaskCreate(TaskBase): pass
class Task(TaskBase):
    id: int; status: str
    result_a: Optional[str] = None; result_b: Optional[str] = None
    class Config: orm_mode = True
