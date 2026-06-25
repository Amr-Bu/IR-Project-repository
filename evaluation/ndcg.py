import math


def dcg(
    relevances
):

    score = 0

    for i, rel in enumerate(
        relevances,
        start=1
    ):

        score += (
            (2 ** rel - 1)
            /
            math.log2(i + 1)
        )

    return score


def ndcg_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
):

    gains = []

    for doc_id in retrieved_docs[:k]:

        gains.append(
            relevant_docs.get(
                doc_id,
                0
            )
        )

    dcg_score = dcg(
        gains
    )

    ideal = sorted(
        relevant_docs.values(),
        reverse=True
    )[:k]

    idcg_score = dcg(
        ideal
    )

    if idcg_score == 0:

        return 0

    return dcg_score / idcg_score