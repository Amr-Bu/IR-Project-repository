import pickle
import numpy as np

from services.embedding.embedding_service import (
    EmbeddingService
)

from evaluation.precision_at_k import (
    precision_at_k
)

from evaluation.recall_at_k import (
    recall_at_k
)

from evaluation.average_precision import (
    average_precision
)

from evaluation.ndcg import (
    ndcg_at_k
)

import faiss


print(
    "Loading ground truth..."
)

with open(
    "data/ground_truth.pkl",
    "rb"
) as f:

    ground_truth = pickle.load(f)


with open(
    "data/full_document_ids.pkl",
    "rb"
) as f:

    document_ids = pickle.load(f)


index = faiss.read_index(
    "data/full_faiss.index"
)

embedding_service = (
    EmbeddingService()
)


import ir_datasets

dataset = ir_datasets.load(
    "beir/webis-touche2020"
)

queries = {
    q.query_id: q.text
    for q in dataset.queries_iter()
}


all_ap = []
all_precision = []
all_recall = []
all_ndcg = []


for counter, (
    query_id,
    relevant_docs
) in enumerate(
    ground_truth.items(),
    start=1
):

    query_text = queries[
        query_id
    ]

    query_embedding = (
        embedding_service.encode(
            [query_text]
        )
    )

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = (
        index.search(
            query_embedding,
            10
        )
    )

    retrieved_docs = []

    for idx in indices[0]:

        retrieved_docs.append(
            document_ids[idx]
        )

    all_ap.append(
        average_precision(
            retrieved_docs,
            relevant_docs
        )
    )

    all_precision.append(
        precision_at_k(
            retrieved_docs,
            relevant_docs,
            10
        )
    )

    all_recall.append(
        recall_at_k(
            retrieved_docs,
            relevant_docs,
            10
        )
    )

    all_ndcg.append(
        ndcg_at_k(
            retrieved_docs,
            relevant_docs,
            10
        )
    )

    print(
        f"Processed Query {counter}/49"
    )


print()
print(
    "===== SEMANTIC RESULTS ====="
)

print(
    f"MAP: {np.mean(all_ap):.4f}"
)

print(
    f"P@10: {np.mean(all_precision):.4f}"
)

print(
    f"Recall@10: {np.mean(all_recall):.4f}"
)

print(
    f"NDCG@10: {np.mean(all_ndcg):.4f}"
)