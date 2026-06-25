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

document_ids = list(
    documents.keys()
)

texts = list(
    documents.values()
)

service = EmbeddingService()

batch_size = 1000

all_embeddings = []

for i in range(
    0,
    len(texts),
    batch_size
):

    batch = texts[
        i:i + batch_size
    ]

    embeddings = service.encode(
        batch
    )

    all_embeddings.append(
        embeddings
    )

    print(
        f"Processed "
        f"{min(i + batch_size, len(texts))}"
        f"/{len(texts)}"
    )

embeddings = np.vstack(
    all_embeddings
)

np.save(
    "data/full_embeddings.npy",
    embeddings
)

with open(
    "data/full_document_ids.pkl",
    "wb"
) as f:

    pickle.dump(
        document_ids,
        f
    )

print(
    embeddings.shape
)

print(
    "Done"
)