import ir_datasets

dataset = ir_datasets.load("beir/webis-touche2020")

doc = next(dataset.docs_iter())

print(type(doc))
print(doc)