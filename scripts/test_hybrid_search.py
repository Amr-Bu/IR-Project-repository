from services.search.hybrid_search_service import (
    HybridSearchService
)

service = (
    HybridSearchService()
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
        document[1]
    )