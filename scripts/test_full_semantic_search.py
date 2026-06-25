from services.search.full_semantic_search_service import (
    FullSemanticSearchService
)

service = (
    FullSemanticSearchService()
)

results = service.search(
    "Should teachers get tenure?"
)

for rank, (
    score,
    document
) in enumerate(
    results,
    start=1
):

    print("\n")
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