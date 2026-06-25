import pickle

with open(
    "data/processed_documents.pkl",
    "rb"
) as f:

    docs = pickle.load(f)

print(type(docs))

print(
    "Number of documents:",
    len(docs)
)

first_key = next(
    iter(docs)
)

print(
    "First key:",
    first_key
)

print(
    "First value:"
)

print(
    docs[first_key]
)