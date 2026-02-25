from sentence_transformers import SentenceTransformer

# Load once at startup
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    embedding = model.encode(text)
    return embedding.astype("float32").tolist()
