from services.retrieval.tfidf_retriever import TFIDFRetriever

from services.database.db_service import DatabaseService


retriever = TFIDFRetriever()

db = DatabaseService()

results = retriever.search(

    "Should schools provide free books to students?",

    top_k=10
)

for rank, (doc_id, score) in enumerate(results, start=1):

    doc = db.get_document_by_id(
        doc_id
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
        doc[1]
    )

    print()

    print(
        "Text:"
    )

    print(
        doc[2][:500]
    )

db.close()