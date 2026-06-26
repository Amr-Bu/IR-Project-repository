from services.topic_detection.topic_detection_service import (
    TopicDetectionService
)

service = (
    TopicDetectionService()
)

queries = [

    "Should homework be banned?",

    "Climate change is dangerous.",

    "Artificial intelligence is replacing jobs.",

    "Should abortion be legal?",

    "Football players earn too much money."

]

for query in queries:

    print()

    print("=" * 80)

    print(query)

    print()

    print(

        service.detect_topic(
            query
        )

    )