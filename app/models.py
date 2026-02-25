from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db import Base


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    prompt_text = Column(Text, nullable=False)
    embedding = Column(JSONB, nullable=False) 
    created_at = Column(DateTime, default=datetime.utcnow)

    queries = relationship("QueryCache", back_populates="prompt")


class QueryCache(Base):
    __tablename__ = "query_cache"

    id = Column(Integer, primary_key=True, index=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"))
    sql_query = Column(Text, nullable=False)
    result_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    prompt = relationship("Prompt", back_populates="queries")