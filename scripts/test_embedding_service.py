from services.embedding.embedding_service import (
    EmbeddingService
)

service = EmbeddingService()

embeddings = service.encode(
    [
        "Should teachers get tenure?",
        "Should schools provide condoms?"
    ]
)

print(
    embeddings.shape
)