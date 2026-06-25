from services.preprocessing.preprocessor import Preprocessor

p = Preprocessor()

text = """
The cats are running quickly
through the gardens.
"""

print(
    p.preprocess(text)
)