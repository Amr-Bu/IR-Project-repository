import pickle
import ir_datasets


dataset = ir_datasets.load(
    "beir/webis-touche2020"
)

ground_truth = {}

for qrel in dataset.qrels_iter():

    query_id = qrel.query_id

    doc_id = qrel.doc_id

    relevance = qrel.relevance

    if relevance <= 0:
        continue

    if query_id not in ground_truth:

        ground_truth[query_id] = {}

    ground_truth[query_id][doc_id] = relevance

print(
    f"Queries: {len(ground_truth)}"
)

with open(
    "data/ground_truth.pkl",
    "wb"
) as f:

    pickle.dump(
        ground_truth,
        f
    )

print(
    "Ground truth saved."
)