from transformers import AutoModel

print("Loading...")

model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("DONE")