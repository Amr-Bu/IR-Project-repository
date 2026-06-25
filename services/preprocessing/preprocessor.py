import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessor:

    def __init__(self):

        self.stop_words = set(
            stopwords.words("english")
        )

        self.lemmatizer = WordNetLemmatizer()

    def preprocess(
        self,
        text
    ):

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        tokens = text.split()

        tokens = [

            self.lemmatizer.lemmatize(token)

            for token in tokens

            if token not in self.stop_words

        ]

        return " ".join(tokens)