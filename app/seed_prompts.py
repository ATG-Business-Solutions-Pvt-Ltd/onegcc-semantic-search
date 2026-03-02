from app.db import SessionLocal
from app.models import Prompt
from app.embeddings import get_embedding
from app.routes import extract_month, extract_city, extract_year

db = SessionLocal()

texts = [
    "What are total sales?",
    "Show overall revenue",
    "Give me total revenue",
    "Sales summary",
    "Total sales for all regions",
    "Sales in January",
    "January revenue",
    "Revenue for Jan",
    "How much did we earn in January?",
    "January total sales",
    "Sales in February",
    "February revenue",
    "Revenue for Feb",
    "February sales report",
    "How much did we earn in February?",
    "Total profit",
    "Overall profit",
    "Show net profit",
    "Profit summary",
    "Company profit report",
    "Sales in 2025",
    "Revenue for 2025",
    "Total earnings in 2025",
    "2025 sales summary",
    "How much did we make in 2025?",
    "Sales in Mumbai",
    "Mumbai revenue",
    "Revenue for Mumbai",
    "Mumbai total sales",
    "Show Mumbai performance"
]

for text in texts:
    existing = db.query(Prompt).filter(Prompt.content == text).first()
    if existing:
        continue
    
    prompt = Prompt(
        content=text,
        prompt_text=text,
        month=extract_month(text),
        city=extract_city(text),
        year=extract_year(text),
        embedding=get_embedding(text)
    )
    db.add(prompt)

db.commit()
db.close()

print("Prompts inserted successfully.")