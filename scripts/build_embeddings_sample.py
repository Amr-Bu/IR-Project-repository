import pickle
import numpy as np

from services.embedding.embedding_service import (
    EmbeddingService
)

with open(
    "data/processed_documents.pkl",
    "rb"
) as f:

    documents = pickle.load(f)

sample_ids = list(
    documents.keys()
)[:10000]

texts = [
    documents[doc_id]
    for doc_id in sample_ids
]

service = EmbeddingService()

embeddings = service.encode(
    texts
)

np.save(
    "data/sample_embeddings.npy",
    embeddings
)

with open(
    "data/sample_document_ids.pkl",
    "wb"
) as f:

    pickle.dump(
        sample_ids,
        f
    )

print(
    embeddings.shape
)

print(
    "Done"
)