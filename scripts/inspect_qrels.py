import ir_datasets

dataset = ir_datasets.load(
    "beir/webis-touche2020"
)

for i, qrel in enumerate(
    dataset.qrels_iter()
):

    print(qrel)

    if i == 20:
        break