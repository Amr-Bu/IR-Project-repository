import sqlite3


class DatabaseService:

    def __init__(self):

        self.conn = sqlite3.connect(
            "database/documents.db",
            check_same_thread=False
            
        )

        self.cursor = self.conn.cursor()

    def create_table(self):

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents(

            doc_id TEXT PRIMARY KEY,

            title TEXT,

            raw_text TEXT,

            stance TEXT,

            url TEXT

        )
        """)

        self.conn.commit()

    def insert_document(
        self,
        doc_id,
        title,
        raw_text,
        stance,
        url
    ):

        self.cursor.execute("""
        INSERT OR REPLACE INTO documents
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            title,
            raw_text,
            stance,
            url
        ))

    def get_document_by_id(
        self,
        doc_id
    ):

        self.cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,)
        )

        return self.cursor.fetchone()

    def commit(self):

        self.conn.commit()

    def close(self):

        self.conn.close()