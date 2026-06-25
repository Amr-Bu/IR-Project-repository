import pickle
import faiss

from services.retrieval.bm25_retriever import (
    BM25Retriever
)

from services.embedding.embedding_service import (
    EmbeddingService
)

from services.database.db_service import (
    DatabaseService
)


class HybridSearchService:

    def __init__(self):

        self.bm25 = BM25Retriever()

        self.embedding_service = (
            EmbeddingService()
        )

        self.db = (
            DatabaseService()
        )

        self.index = faiss.read_index(
            "data/full_faiss.index"
        )

        with open(
            "data/full_document_ids.pkl",
            "rb"
        ) as f:

            self.document_ids = (
                pickle.load(f)
            )

    def search(
        self,
        query,
        top_k=10,
        alpha=0.7
    ):

        bm25_results = (
            self.bm25.search(
                query,
                top_k=100
            )
        )

        bm25_scores = {}

        for doc_id, score in bm25_results:

            bm25_scores[
                doc_id
            ] = score

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
                100
            )
        )

        semantic_scores = {}

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            semantic_scores[
                self.document_ids[idx]
            ] = float(score)

        all_docs = (
            set(bm25_scores.keys())
            |
            set(semantic_scores.keys())
        )

        max_bm25 = (
            max(bm25_scores.values())
            if bm25_scores
            else 1
        )

        final_scores = []

        for doc_id in all_docs:

            bm25_score = (
                bm25_scores.get(
                    doc_id,
                    0
                )
                /
                max_bm25
            )

            semantic_score = (
                semantic_scores.get(
                    doc_id,
                    0
                )
            )

            final_score = (
                alpha * bm25_score
                +
                (1 - alpha)
                * semantic_score
            )

            final_scores.append(
                (
                    doc_id,
                    final_score
                )
            )

        final_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for doc_id, score in final_scores[:top_k]:

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