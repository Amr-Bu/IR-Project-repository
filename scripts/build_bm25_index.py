import pickle

from rank_bm25 import BM25Okapi


print("Loading processed documents...")

with open(
    "data/processed_documents.pkl",
    "rb"
) as f:

    docs = pickle.load(f)

doc_ids = list(
    docs.keys()
)

tokenized_docs = []

count = 0

for text in docs.values():

    tokenized_docs.append(
        text.split()
    )

    count += 1

    if count % 50000 == 0:

        print(
            f"{count} documents loaded"
        )

print()

print("Building BM25...")

bm25 = BM25Okapi(
    tokenized_docs
)

with open(
    "indexes/bm25_index.pkl",
    "wb"
) as f:

    pickle.dump(
        bm25,
        f
    )

with open(
    "indexes/bm25_doc_ids.pkl",
    "wb"
) as f:

    pickle.dump(
        doc_ids,
        f
    )

print()

print("BM25 index saved.")