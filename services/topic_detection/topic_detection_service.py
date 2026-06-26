import pickle

from services.embedding.embedding_service import (
    EmbeddingService
)


class TopicDetectionService:

    TOPICS = {

        0: "Education",

        1: "Politics",

        2: "Technology",

        3: "Health",

        4: "Environment",

        5: "Society",

        6: "Economy",

        7: "Sports",

        8: "Science",

        9: "General"

    }

    def __init__(self):

        with open(
            "data/topic_model.pkl",
            "rb"
        ) as f:

            self.model = pickle.load(
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