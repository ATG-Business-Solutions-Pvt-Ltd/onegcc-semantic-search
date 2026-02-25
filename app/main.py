from app.faiss_index import build_index
from app.routes import router

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from app import models
from app.schemas import PromptCreate, SearchQuery
from app.embeddings import get_embedding
from app.vector_store import add_vector, search
from sqlalchemy.exc import OperationalError
import time

from app.db import SessionLocal, engine
from app.models import Base
from app.models import Prompt, QueryCache

models = Base.metadata.create_all(bind=engine)
app = FastAPI()
app.include_router(router)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup():
    retries = 5
    while retries:
        try:
            Base.metadata.create_all(bind=engine)
            print("Database connected successfully")
            break
        except OperationalError:
            print("Database not ready, retrying...")
            retries -= 1
            time.sleep(5)

    db = SessionLocal()
    prompts = db.query(Prompt).all()
    build_index(prompts)
    print(f"Built FAISS index with {len(prompts)} prompts")
    db.close()

@app.post("/add")
def add_prompt(prompt: PromptCreate, db: Session = Depends(get_db)):
    embedding = get_embedding(prompt.content)
    db_prompt = Prompt(content=prompt.content, embedding=embedding)
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)

    vector = get_embedding(prompt.content)
    add_vector(vector, db_prompt.id)

    return {"message": "Prompt added", "id": db_prompt.id}

@app.post("/search")
def search_prompt(query: SearchQuery):
    vector = get_embedding(query.query)
    results = search(vector)

    return {"matching_prompt_ids": results}