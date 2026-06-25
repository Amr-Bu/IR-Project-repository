import pickle
import numpy as np

from services.embedding.embedding_service import (
    EmbeddingService
)

print(
    "Loading processed documents..."
)

with open(
    "data/processed_documents.pkl",
    "rb"
) as f:

    documents = pickle.load(f)

print(
    f"Loaded {len(documents)} documents"
)

service = EmbeddingService()

document_ids = []
all_embeddings = []

batch_size = 1000

for start in range(
    0,
    len(documents),
    batch_size
):

    end = min(
        start + batch_size,
        len(documents)
    )

    batch_docs = documents[
        start:end
    ]

    texts = [
        doc["text"]
        for doc in batch_docs
    ]

    embeddings = service.encode(
        texts
    )

    all_embeddings.append(
        embeddings
    )

    document_ids.extend(
        [
            doc["doc_id"]
            for doc in batch_docs
        ]
    )

    print(
        f"Processed {end}/{len(documents)}"
    )

embeddings = np.vstack(
    all_embeddings
)

np.save(
    "data/document_embeddings.npy",
    embeddings
)

with open(
    "data/document_ids.pkl",
    "wb"
) as f:

    pickle.dump(
        document_ids,
        f
    )

print(
    "Embeddings saved."
)