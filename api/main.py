from fastapi import FastAPI

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
    title="IR Search Engine",
    version="1.0"
)

bm25 = BM25Retriever()

semantic = (
    FullSemanticSearchService()
)

hybrid = (
    HybridSearchService()
)


@app.get("/")
def home():

    return {
        "message": "IR Search Engine API"
    }


@app.get("/search/bm25")
def search_bm25(
    query: str,
    top_k: int = 10
):

    results = bm25.search(
        query,
        top_k
    )

    output = []

    for doc_id, score in results:

        output.append(
            { "doc_id": document[0],
                "title": document[1],
                "stance": document[3],
                "score": float(score)
            }
        )

    return output


@app.get("/search/semantic")
def search_semantic(
    query: str,
    top_k: int = 10
):

    results = semantic.search(
        query,
        top_k
    )

    output = []

    for score, document in results:

        output.append(
            {
                "doc_id": document[0],
                "title": document[1],
                "stance": document[3],
                "score": float(score)
            }
        )

    return output


@app.get("/search/hybrid")
def search_hybrid(
    query: str,
    top_k: int = 10
):

    results = hybrid.search(
        query,
        top_k
    )

    output = []

    for score, document in results:

        output.append(
            {
                "doc_id": document[0],
                "title": document[1],
                "stance": document[3],
                "score": float(score)
            }
        )

    return output