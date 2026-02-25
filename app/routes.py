from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas import AskRequest
from app.db import SessionLocal
from app.models import Prompt, QueryCache
import json
from app.embeddings import get_embedding
from app.faiss_index import add_to_index, search_index
import numpy as np

router = APIRouter()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/ask")
def ask_question(request: AskRequest, db: Session = Depends(get_db)):

    query_embedding = get_embedding(request.prompt)
    similar_ids = search_index(query_embedding, k=1)

    if similar_ids:
        similar_prompt = db.query(Prompt).filter(Prompt.id == similar_ids[0]).first()

        if similar_prompt:
            cached_query = db.query(QueryCache).filter(
                QueryCache.prompt_id == similar_prompt.id
            ).first()

            return {
                "source": "semantic_match",
                "matched_prompt": similar_prompt.content,
            }
            
    generated_sql = f"SELECT * FROM sales WHERE question = '{request.prompt}'"
    fake_result = {"data": "Sample result"}

    new_prompt = Prompt(content=request.prompt, prompt_text=request.prompt, embedding=query_embedding)
    db.add(new_prompt)
    db.commit()
    db.refresh(new_prompt)

    add_to_index(new_prompt.id, query_embedding)

    new_query = QueryCache(
        prompt_id=new_prompt.id,
        sql_query=generated_sql,
        result_json=fake_result
    )
    db.add(new_query)
    db.commit()

    return {
        "source": "llm",
        "sql_query": generated_sql,
        "result": fake_result
    }
