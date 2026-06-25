import ir_datasets

dataset = ir_datasets.load("beir/webis-touche2020")

print("Loading dataset...")

docs_count = sum(1 for _ in dataset.docs_iter())
queries_count = sum(1 for _ in dataset.queries_iter())
qrels_count = sum(1 for _ in dataset.qrels_iter())

print(f"Documents: {docs_count}")
print(f"Queries: {queries_count}")
print(f"Qrels: {qrels_count}")