import pickle

with open(
    "data/processed_documents.pkl",
    "rb"
) as f:

    docs = pickle.load(f)

print(
    "Documents:",
    len(docs)
)

first_key = next(
    iter(docs)
)

print()

print(
    "Doc ID:",
    first_key
)

print()

print(
    docs[first_key][:500]
)