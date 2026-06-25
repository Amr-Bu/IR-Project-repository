import faiss
import numpy as np

embeddings = np.load(
    "data/sample_embeddings.npy"
)

embeddings = embeddings.astype(
    "float32"
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

faiss.normalize_L2(
    embeddings
)

index.add(
    embeddings
)

faiss.write_index(
    index,
    "data/sample_faiss.index"
)

print(
    "Vectors:",
    index.ntotal
)