import faiss
import numpy as np

embeddings = np.load(
    "data/full_embeddings.npy"
)

print(
    "Embeddings:",
    embeddings.shape
)

index = faiss.IndexFlatIP(
    embeddings.shape[1]
)

faiss.normalize_L2(
    embeddings
)

index.add(
    embeddings
)

faiss.write_index(
    index,
    "data/full_faiss.index"
)

print(
    "FAISS saved."
)