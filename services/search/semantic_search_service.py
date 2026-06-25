import pickle
import faiss
import numpy as np

from services.embedding.embedding_service import (
    EmbeddingService
)


class SemanticSearchService:

    def __init__(self):

        self.model = EmbeddingService()

        self.index = faiss.read_index(
            "data/sample_faiss.index"
        )

        with open(
            "data/sample_document_ids.pkl",
            "rb"
        ) as f:

            self.document_ids = pickle.load(f)

    def search(
        self,
        query,
        top_k=10
    ):

        query_embedding = (
            self.model.encode(
                [query]
            )
        )

        query_embedding = (
            query_embedding.astype(
                "float32"
            )
        )

        faiss.normalize_L2(
            query_embedding
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            results.append(
                (
                    self.document_ids[idx],
                    float(score)
                )
            )

        return results