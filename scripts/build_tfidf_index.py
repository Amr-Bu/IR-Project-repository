import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy import sparse


print("Loading processed documents...")

with open(
    "data/processed_documents.pkl",
    "rb"
) as f:

    docs = pickle.load(f)

doc_ids = list(
    docs.keys()
)

documents = list(
    docs.values()
)

print(
    f"Loaded {len(documents)} documents"
)

print("Building TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(
    documents
)

print(
    tfidf_matrix.shape
)

with open(
    "indexes/tfidf_vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f
    )

with open(
    "indexes/doc_ids.pkl",
    "wb"
) as f:

    pickle.dump(
        doc_ids,
        f
    )

sparse.save_npz(
    "indexes/tfidf_matrix.npz",
    tfidf_matrix
)

print("TF-IDF index saved.")