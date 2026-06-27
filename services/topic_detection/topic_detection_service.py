import pickle

from services.embedding.embedding_service import (
    EmbeddingService
)


class TopicDetectionService:

    TOPICS = {
        0: "Philosophy & Theology",
        1: "Human Rights & Society",
        2: "Entertainment & Pop Culture",
        3: "Society & Debate Culture",
        4: "Religion, Science & Fringe Topics",
        5: "Education",
        6: "Religion",
        7: "Bioethics & Ethics",
        8: "General / Mixed Topics",
        9: "General & Miscellaneous"
    }

    def __init__(self):

        with open(
            "data/topic_model.pkl",
            "rb"
        ) as f:

            self.model = pickle.load(
                f
            )

        with open(
            "data/document_topic_map.pkl",
            "rb"
        ) as f:

            self.document_topics = pickle.load(
                f
            )

        self.embedding = (
            EmbeddingService()
        )

    def detect_topic(
        self,
        query
    ):

        vector = (
            self.embedding.encode(
                [query]
            )
        )

        cluster = int(

            self.model.predict(
                vector
            )[0]

        )

        return {

            "cluster": cluster,

            "topic": self.TOPICS.get(
                cluster,
                "Unknown"
            )

        }

    def get_document_topic(
        self,
        doc_id
    ):

        cluster = self.document_topics.get(
            doc_id
        )

        if cluster is None:
            return "Unknown"

        return self.TOPICS.get(
            cluster,
            "Unknown"
        )