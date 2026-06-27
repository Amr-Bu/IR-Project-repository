import pickle

from services.database.db_service import (
    DatabaseService
)

NUM_CLUSTERS = 10
SAMPLES_PER_CLUSTER = 10

print("Loading document topics...")
with open("data/document_topics.pkl", "rb") as f:
    labels = pickle.load(f)

print("Loading document ids...")
with open("data/full_document_ids.pkl", "rb") as f:
    document_ids = pickle.load(f)

print(f"Total documents: {len(document_ids)}")
print()

db = DatabaseService()

clusters = {i: [] for i in range(NUM_CLUSTERS)}
cluster_sizes = {i: 0 for i in range(NUM_CLUSTERS)}

for doc_id, label in zip(document_ids, labels):
    cluster_sizes[label] += 1
    if len(clusters[label]) < SAMPLES_PER_CLUSTER:
        clusters[label].append(doc_id)

for cluster_id in range(NUM_CLUSTERS):
    print("=" * 80)
    print(f"Cluster {cluster_id}  ({cluster_sizes[cluster_id]} documents)")
    print("=" * 80)
    for doc_id in clusters[cluster_id]:
        document = db.get_document_by_id(doc_id)
        if document is None:
            continue
        print(f"  - {document[1]}")
    print()

db.close()
print("Done.")