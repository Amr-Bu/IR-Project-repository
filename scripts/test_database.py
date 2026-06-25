import sqlite3

conn = sqlite3.connect(
    "database/documents.db"
)

cursor = conn.cursor()

cursor.execute("""
SELECT *
FROM documents
LIMIT 3
""")

rows = cursor.fetchall()

for row in rows:

    print("DOC ID:")
    print(row[0])

    print()

    print("TITLE:")
    print(row[1])

    print()

    print("TEXT:")

    print(
        row[2][:500]
    )

    print()

    print("=" * 100)

conn.close()