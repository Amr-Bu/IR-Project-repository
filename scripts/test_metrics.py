from evaluation.precision_at_k import precision_at_k
from evaluation.recall_at_k import recall_at_k
from evaluation.average_precision import average_precision
from evaluation.ndcg import ndcg_at_k


retrieved = [
    "A",
    "B",
    "C",
    "D",
    "E"
]

relevant = {
    "A": 5,
    "C": 4,
    "F": 3
}

print(
    "Precision:",
    precision_at_k(
        retrieved,
        relevant,
        5
    )
)

print(
    "Recall:",
    recall_at_k(
        retrieved,
        relevant,
        5
    )
)

print(
    "AP:",
    average_precision(
        retrieved,
        relevant
    )
)

print(
    "NDCG:",
    ndcg_at_k(
        retrieved,
        relevant,
        5
    )
)