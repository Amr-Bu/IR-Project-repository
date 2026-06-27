from nltk.corpus import wordnet


class QueryRefinementService:

    def __init__(
        self,
        max_synonyms_per_word=2
    ):

        self.max_synonyms_per_word = (
            max_synonyms_per_word
        )

    def get_synonyms(
        self,
        word
    ):

        synonyms = set()

        for syn in wordnet.synsets(word):

            for lemma in syn.lemmas():

                candidate = (
                    lemma.name()
                    .replace("_", " ")
                    .lower()
                )

                if candidate != word.lower():

                    synonyms.add(candidate)

        return list(synonyms)[
            : self.max_synonyms_per_word
        ]

    def expand_query(
        self,
        query
    ):

        words = query.split()

        expanded_terms = list(words)

        for word in words:

            clean_word = (
                word.strip(".,?!").lower()
            )

            synonyms = self.get_synonyms(
                clean_word
            )

            expanded_terms.extend(
                synonyms
            )

        return " ".join(expanded_terms)