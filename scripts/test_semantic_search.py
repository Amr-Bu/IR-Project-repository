from services.search.semantic_search_service import (
    SemanticSearchService
)

from services.database.db_service import (
    DatabaseService
)

service = (
    SemanticSearchService()
)

db = (
    DatabaseService()
)

results = service.search(
    "Should teachers get tenure?"
)

for rank, (
    doc_id,
    score
) in enumerate(
    results,
    start=1
):

    document = (
        db.get_document_by_id(
            doc_id
        )
    )

    print()

    print("=" * 100)

    print(
        f"Rank: {rank}"
    )

    print(
        f"Score: {score}"
    )

    print()

    print(
        "Title:"
    )

    print(
        document[1]
    )

    print()

    print(
        "Text:"
    )

    print(
        document[2][:500]
    )