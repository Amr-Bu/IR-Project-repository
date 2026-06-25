from sentence_transformers import SentenceTransformer

print(
    "Loading model..."
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embedding = model.encode(
    "Should teachers get tenure?"
)

print(
    embedding.shape
)