import faiss
import numpy as np

DIMENSION = 384

# Global FAISS index
index = faiss.IndexFlatL2(DIMENSION)

# Mapping FAISS position → DB id
id_map = []


def build_index(prompts):
    global index, id_map

    # Reset index
    index = faiss.IndexFlatL2(DIMENSION)
    id_map = []

    if not prompts:
        return

    embeddings = np.array(
        [p.embedding for p in prompts],
        dtype="float32"
    )

    index.add(embeddings)
    id_map = [p.id for p in prompts]


def add_to_index(prompt_id, embedding):
    global id_map

    vector = np.array([embedding], dtype="float32")
    index.add(vector)
    id_map.append(prompt_id)


def search_index(query_embedding, k=10):
    if index.ntotal == 0:
        return [], []

    query_vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vector, k)

    matched_ids = []
    matched_distances = []

    for i, idx in enumerate(indices[0]):
        if idx < len(id_map):
            matched_ids.append(id_map[idx])
            matched_distances.append(distances[0][i])

    return matched_ids, matched_distances