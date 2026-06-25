import pickle
import ir_datasets

from services.retrieval.tfidf_retriever import TFIDFRetriever

from evaluation.precision_at_k import precision_at_k
from evaluation.recall_at_k import recall_at_k
from evaluation.average_precision import average_precision
from evaluation.ndcg import ndcg_at_k


print("Loading ground truth...")

with open(
    "data/ground_truth.pkl",
    "rb"
) as f:

    ground_truth = pickle.load(f)

dataset = ir_datasets.load(
    "beir/webis-touche2020"
)

retriever = TFIDFRetriever()

precision_scores = []
recall_scores = []
ap_scores = []
ndcg_scores = []

query_count = 0

for query in dataset.queries_iter():

    query_id = query.query_id

    if query_id not in ground_truth:
        continue

    results = retriever.search(
        query.text,
        top_k=10
    )

    retrieved_docs = [
        doc_id
        for doc_id, score in results
    ]

    relevant_docs = ground_truth[
        query_id
    ]

    precision_scores.append(
        precision_at_k(
            retrieved_docs,
            relevant_docs,
            10
        )
    )

    recall_scores.append(
        recall_at_k(
            retrieved_docs,
            relevant_docs,
            10
        )
    )

    ap_scores.append(
        average_precision(
            retrieved_docs,
            relevant_docs
        )
    )

    ndcg_scores.append(
        ndcg_at_k(
            retrieved_docs,
            relevant_docs,
            10
        )
    )

    query_count += 1

    print(
        f"Processed Query {query_count}/49"
    )

print("\n===== TF-IDF RESULTS =====")

print(
    f"MAP: {sum(ap_scores)/len(ap_scores):.4f}"
)

print(
    f"P@10: {sum(precision_scores)/len(precision_scores):.4f}"
)

print(
    f"Recall@10: {sum(recall_scores)/len(recall_scores):.4f}"
)

print(
    f"NDCG@10: {sum(ndcg_scores)/len(ndcg_scores):.4f}"
)