from fastapi import FastAPI

from services.database.db_service import (
    DatabaseService
)

from services.retrieval.tfidf_retriever import (
    TFIDFRetriever
)

from services.retrieval.bm25_retriever import (
    BM25Retriever
)

from services.search.full_semantic_search_service import (
    FullSemanticSearchService
)

from services.search.hybrid_search_service import (
    HybridSearchService
)


app = FastAPI(
    title="IR Search API"
)


db = DatabaseService()

tfidf = TFIDFRetriever()

bm25 = BM25Retriever()

semantic = FullSemanticSearchService()

hybrid = HybridSearchService()


@app.get("/")
def home():

    return {
        "message": "IR Search API is running"
    }


@app.get("/search/tfidf")
def search_tfidf(query: str):

    results = tfidf.search(
        query,
        top_k=10
    )

    output = []

    for doc_id, score in results:

        document = db.get_document_by_id(
            doc_id
        )

        if document is None:
            continue

        output.append(
            {
                "doc_id": document[0],
                "title": document[1],
                "text": document[2][:100] + "...",
                "stance": document[3],
                "score": float(score)
            }
        )

    return output


@app.get("/search/bm25")
def search_bm25(query: str):

    results = bm25.search(
        query,
        top_k=10
    )

    output = []

    for doc_id, score in results:

        document = db.get_document_by_id(
            doc_id
        )

        if document is None:
            continue

        output.append(
            {
                "doc_id": document[0],
                "title": document[1],
                "text": document[2][:100] + "...",
                "stance": document[3],
                "score": float(score)
            }
        )

    return output


@app.get("/search/semantic")
def search_semantic(query: str):

    results = semantic.search(
        query,
        top_k=10
    )

    output = []

    for score, document in results:

        output.append(
            {
                "doc_id": document[0],
                "title": document[1],
                "text": document[2][:100] + "...",
                "stance": document[3],
                "score": float(score)
            }
        )

    return output


@app.get("/search/hybrid")
def search_hybrid(query: str):

    results = hybrid.search(
        query,
        top_k=10
    )

    output = []

    for score, document in results:

        output.append(
            {
                "doc_id": document[0],
                "title": document[1],
                "text": document[2][:100] + "...",
                "stance": document[3],
                "score": float(score)
            }
        )

    return output