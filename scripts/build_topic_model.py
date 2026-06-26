import pickle

import numpy as np

from sklearn.cluster import KMeans


print("Loading embeddings...")

embeddings = np.load(
    "data/full_embeddings.npy"
)

print(
    embeddings.shape
)

print()

print("Training Topic Model...")

kmeans = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(
    embeddings
)

with open(
    "data/topic_model.pkl",
    "wb"
) as f:

    pickle.dump(
        kmeans,
        f
    )

with open(
    "data/document_topics.pkl",
    "wb"
) as f:

    pickle.dump(
        labels,
        f
    )

print()

print("Done.")