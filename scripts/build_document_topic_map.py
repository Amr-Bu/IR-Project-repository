import pickle


print("Loading document ids...")
with open("data/full_document_ids.pkl", "rb") as f:
    document_ids = pickle.load(f)

print("Loading document topics...")
with open("data/document_topics.pkl", "rb") as f:
    labels = pickle.load(f)

document_topic_map = {
    doc_id: int(label)
    for doc_id, label in zip(document_ids, labels)
}

with open("data/document_topic_map.pkl", "wb") as f:
    pickle.dump(document_topic_map, f)

print(f"Saved mapping for {len(document_topic_map)} documents.")
print("Done.")