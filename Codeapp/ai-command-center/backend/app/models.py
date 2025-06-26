from sqlalchemy import Column, Integer, String, Text, DateTime, func
from .database import Base
class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String, nullable=False); status = Column(String, default="PENDING")
    result_a = Column(Text); result_b = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
