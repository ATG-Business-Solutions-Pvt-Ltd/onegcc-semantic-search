from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas import AskRequest
from app.db import SessionLocal
from app.models import Prompt, QueryCache
from app.embeddings import get_embedding
from app.faiss_index import add_to_index, search_index
import re

CITIES = ["mumbai","delhi","hyderabad","bangalore","chennai"]
MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december"]

router = APIRouter()


def extract_city(text):
    text = text.lower()
    for city in CITIES:
        if city in text:
            return city
    return None


def extract_month(text):
    text = text.lower()
    for month in MONTHS:
        if month in text:
            return month
    return None


def extract_year(text):
    match = re.search(r'\b(20\d{2}|19\d{2})\b', text)
    return match.group(1) if match else None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/ask")
def ask_question(request: AskRequest, db: Session = Depends(get_db)):

    query_embedding = get_embedding(request.prompt)

    matched_ids, distances = search_index(query_embedding, k=5)

    city = extract_city(request.prompt)
    month = extract_month(request.prompt)
    year = extract_year(request.prompt)

    semantic_threshold = 1.3

    # 🔹 Apply threshold + metadata filtering AFTER vector search
    for i, prompt_id in enumerate(matched_ids):

        distance = distances[i]

        if distance > semantic_threshold:
            continue

        similar_prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()

        if not similar_prompt:
            continue

        content_lower = similar_prompt.content.lower()

        if city and city not in content_lower:
            continue

        if month and month not in content_lower:
            continue

        if year and year not in content_lower:
            continue

        cached_query = db.query(QueryCache).filter(
            QueryCache.prompt_id == similar_prompt.id
        ).first()

        if cached_query:
            return {
                "source": "semantic_cache",
                "matched_prompt": similar_prompt.content,
                "distance": float(distance),
                "sql_query": cached_query.sql_query,
                "result": cached_query.result_json
            }

    # 🔹 LLM fallback
    generated_sql = f"SELECT * FROM sales WHERE question = '{request.prompt}'"
    fake_result = {"data": "Sample result"}

    new_prompt = Prompt(
        content=request.prompt,
        prompt_text=request.prompt,
        embedding=query_embedding
    )

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