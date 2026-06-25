import pickle

from services.preprocessing.preprocessor import Preprocessor


class BM25Retriever:

    def __init__(self):

        with open(
            "indexes/bm25_index.pkl",
            "rb"
        ) as f:

            self.bm25 = pickle.load(f)

        with open(
            "indexes/bm25_doc_ids.pkl",
            "rb"
        ) as f:

            self.doc_ids = pickle.load(f)

        self.preprocessor = Preprocessor()

    def search(
        self,
        query,
        top_k=10
    ):

        query = self.preprocessor.preprocess(
            query
        )

        query_tokens = query.split()

        scores = self.bm25.get_scores(
            query_tokens
        )

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []

        for idx in top_indices:

            results.append(
                (
                    self.doc_ids[idx],
                    float(scores[idx])
                )
            )

        return results