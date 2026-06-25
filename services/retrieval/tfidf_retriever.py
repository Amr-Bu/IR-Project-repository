import pickle

from scipy import sparse

from sklearn.metrics.pairwise import cosine_similarity

from services.preprocessing.preprocessor import Preprocessor


class TFIDFRetriever:

    def __init__(self):

        with open(
            "indexes/tfidf_vectorizer.pkl",
            "rb"
        ) as f:

            self.vectorizer = pickle.load(f)

        with open(
            "indexes/doc_ids.pkl",
            "rb"
        ) as f:

            self.doc_ids = pickle.load(f)

        self.matrix = sparse.load_npz(
            "indexes/tfidf_matrix.npz"
        )

        self.preprocessor = Preprocessor()

    def search(
        self,
        query,
        top_k=10
    ):

        query = self.preprocessor.preprocess(
            query
        )

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

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