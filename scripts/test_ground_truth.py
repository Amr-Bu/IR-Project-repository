import pickle

with open(
    "data/ground_truth.pkl",
    "rb"
) as f:

    gt = pickle.load(f)

print(
    "Queries:",
    len(gt)
)

first_query = next(
    iter(gt)
)

print()

print(
    "Query ID:",
    first_query
)

print()

for doc_id, rel in list(
    gt[first_query].items()
)[:10]:

    print(
        doc_id,
        rel
    )
    