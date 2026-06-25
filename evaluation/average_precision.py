def average_precision(
    retrieved_docs,
    relevant_docs
):

    score = 0.0

    hits = 0

    for i, doc_id in enumerate(
        retrieved_docs,
        start=1
    ):

        if doc_id in relevant_docs:

            hits += 1

            score += hits / i

    if hits == 0:

        return 0

    return score / hits