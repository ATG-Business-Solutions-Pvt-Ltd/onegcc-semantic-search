from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas import AskRequest
from app.db import SessionLocal
from app.models import Prompt, QueryCache
from app.embeddings import get_embedding

import re
import calendar
import spacy
import numpy as np
import faiss

# -------------------------
# Config
# -------------------------

DIMENSION = 384
SEMANTIC_THRESHOLD = 0.75

nlp = spacy.load("en_core_web_sm")
router = APIRouter()


# -------------------------
# Metadata Extractors
# -------------------------

def extract_city(text: str):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text.lower()
    return None


def extract_month(text: str):
    text = text.lower()
    months = [m.lower() for m in calendar.month_name if m]
    for month in months:
        if month in text:
            return month
    return None


def extract_year(text: str):
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    return match.group(1) if match else None


# -------------------------
# DB Dependency
# -------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------
# Main Ask Route
# -------------------------

@router.post("/ask")
def ask_question(request: AskRequest, db: Session = Depends(get_db)):

    query_text = request.prompt
    query_embedding = get_embedding(query_text)

    # Extract metadata
    city = extract_city(query_text)
    month = extract_month(query_text)
    year = extract_year(query_text)

    print("Query:", query_text)
    print("Extracted -> City:", city, "Month:", month, "Year:", year)

    # -------------------------
    # 1️⃣ Metadata Pre-Filtering
    # -------------------------

    query = db.query(Prompt)

    if city:
        query = query.filter(Prompt.city == city)

    if month:
        query = query.filter(Prompt.month == month)

    if year:
        query = query.filter(Prompt.year == year)

    filtered_prompts = query.all()

    # If no structured match found → fallback to global search
    if not filtered_prompts:
        print("No metadata match found. Falling back to global search.")
        filtered_prompts = db.query(Prompt).all()

    if not filtered_prompts:
        print("No prompts in database.")
        return {
            "source": "llm",
            "sql_query": "No prompts available",
            "result": {"data": "No data"}
        }

    # -------------------------
    # 2️⃣ Build Temporary FAISS (Cosine Similarity)
    # -------------------------

    embeddings = np.array(
        [p.embedding for p in filtered_prompts],
        dtype="float32"
    )

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings)

    query_vector = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(query_vector)

    similarities, indices = index.search(
        query_vector,
        k=len(filtered_prompts)
    )

    # -------------------------
    # 3️⃣ Threshold + Cache Check
    # -------------------------

    for i, idx in enumerate(indices[0]):

        similarity = similarities[0][i]
        candidate_prompt = filtered_prompts[idx]

        print("Similarity:", similarity, "| Prompt:", candidate_prompt.content)

        if similarity < SEMANTIC_THRESHOLD:
            continue

        cached_query = db.query(QueryCache).filter(
            QueryCache.prompt_id == candidate_prompt.id
        ).first()

        if cached_query:
            print("Returning semantic cache result.")
            return {
                "source": "semantic_cache",
                "matched_prompt": candidate_prompt.content,
                "similarity": float(similarity),
                "sql_query": cached_query.sql_query,
                "result": cached_query.result_json
            }

    # -------------------------
    # 4️⃣ LLM Fallback
    # -------------------------

    print("Falling back to LLM.")

    generated_sql = f"SELECT * FROM sales WHERE question = '{query_text}'"
    fake_result = {"data": "Sample result"}

    new_prompt = Prompt(
        content=query_text,
        prompt_text=query_text,
        month=month,
        city=city,
        year=year,
        embedding=query_embedding
    )

    db.add(new_prompt)
    db.commit()
    db.refresh(new_prompt)

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