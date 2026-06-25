from services.retrieval.bm25_retriever import BM25Retriever

retriever = BM25Retriever()

results = retriever.search(
    "Should schools provide free condoms to students?",
    top_k=10
)

for doc_id, score in results:

    print(
        doc_id,
        score
    )