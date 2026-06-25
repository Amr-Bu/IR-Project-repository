import pickle

with open(
    "data/ground_truth.pkl",
    "rb"
) as f:

    ground_truth = pickle.load(f)

print(type(ground_truth))

first_key = next(
    iter(ground_truth)
)

print()

print("FIRST KEY:")
print(first_key)

print()

print("FIRST VALUE:")
print(
    ground_truth[first_key]
)