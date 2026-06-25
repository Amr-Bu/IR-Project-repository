def recall_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
):

    retrieved_docs = retrieved_docs[:k]

    hits = 0

    for doc_id in retrieved_docs:

        if doc_id in relevant_docs:

            hits += 1

    return hits / len(relevant_docs)