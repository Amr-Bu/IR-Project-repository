import pickle
import faiss
import numpy as np

from services.embedding.embedding_service import (
    EmbeddingService
)

from services.database.db_service import (
    DatabaseService
)


class FullSemanticSearchService:

    def __init__(self):

        self.embedding_service = (
            EmbeddingService()
        )

        self.index = faiss.read_index(
            "data/full_faiss.index"
        )

        with open(
            "data/full_document_ids.pkl",
            "rb"
        ) as f:

            self.document_ids = pickle.load(f)

        self.db = DatabaseService()

    def search(
        self,
        query,
        top_k=10
    ):

        query_embedding = (
            self.embedding_service.encode(
                [query]
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

            doc_id = (
                self.document_ids[idx]
            )

            document = (
                self.db.get_document_by_id(
                    doc_id
                )
            )

            results.append(
                (
                    score,
                    document
                )
            )

        return results