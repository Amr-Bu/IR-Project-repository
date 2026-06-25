import ir_datasets

dataset = ir_datasets.load(
    "beir/webis-touche2020"
)

for i, query in enumerate(
    dataset.queries_iter()
):

    print()

    print(query)

    if i == 9:
        break