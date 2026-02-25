import faiss
import numpy as np

# Embedding dimension for all-MiniLM-L6-v2
DIMENSION = 384

# In-memory FAISS index
index = faiss.IndexFlatL2(DIMENSION)

# Store mapping from vector position → prompt id
id_map = []


def add_vector(vector: np.ndarray, prompt_id: int):
    global index, id_map
    index.add(np.array([vector]))
    id_map.append(prompt_id)


def search(vector: np.ndarray, top_k: int = 5):
    global index, id_map

    if index.ntotal == 0:
        return []

    distances, indices = index.search(np.array([vector]), top_k)

    results = []
    for idx in indices[0]:
        if idx < len(id_map):
            results.append(id_map[idx])

    return results