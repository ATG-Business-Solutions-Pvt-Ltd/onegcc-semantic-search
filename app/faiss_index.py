import faiss
import numpy as np
from app import models

DIMENSION = 384

# L2 index (we can switch to cosine later)
index = faiss.IndexFlatL2(DIMENSION)

# Keeps mapping from FAISS position → DB id
id_map = []

def build_index(prompts):
    global id_map
    
    if not prompts:
        return

    embeddings = np.array(
        [p.embedding for p in prompts],
        dtype="float32"
    )

    index.add(embeddings)
    id_map = [p.id for p in prompts]


def add_to_index(prompt_id, embedding):
    vector = np.array([embedding], dtype="float32")
    index.add(vector)
    id_map.append(prompt_id)


def search_index(query_embedding, k=3):
    if index.ntotal == 0:
        return []

    query_vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vector, k)

    best_distance = distances[0][0]
    best_index = indices[0][0]

    if best_index < len(id_map):
        return id_map[best_index], best_distance
    
    return None, None