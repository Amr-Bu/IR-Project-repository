from services.retrieval.tfidf_retriever import TFIDFRetriever

retriever = TFIDFRetriever()

results = retriever.search(

    "Should schools provide free condoms to students?",

    top_k=10
)

for doc_id, score in results:

    print(
        doc_id,
        score
    )