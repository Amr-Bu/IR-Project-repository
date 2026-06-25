import ir_datasets

from services.database.db_service import DatabaseService


dataset = ir_datasets.load(
    "beir/webis-touche2020"
)

db = DatabaseService()

db.create_table()

count = 0

for doc in dataset.docs_iter():

    db.insert_document(

        doc_id=doc.doc_id,

        title=doc.title,

        raw_text=doc.text,

        stance=doc.stance,

        url=doc.url
    )

    count += 1

    if count % 5000 == 0:

        db.commit()

        print(
            f"{count} documents inserted"
        )

db.commit()

db.close()

print(
    f"Finished: {count}"
)