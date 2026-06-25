import sqlite3
import pickle

from services.preprocessing.preprocessor import Preprocessor


conn = sqlite3.connect(
    "database/documents.db"
)

cursor = conn.cursor()

cursor.execute("""
SELECT doc_id, raw_text
FROM documents
""")

preprocessor = Preprocessor()

processed_documents = {}

count = 0

for doc_id, raw_text in cursor:

    clean_text = preprocessor.preprocess(
        raw_text
    )

    processed_documents[doc_id] = clean_text

    count += 1

    if count % 5000 == 0:

        print(
            f"{count} documents processed"
        )

conn.close()

with open(
    "data/processed_documents.pkl",
    "wb"
) as f:

    pickle.dump(
        processed_documents,
        f
    )

print(
    f"\nFinished processing {count} documents"
)