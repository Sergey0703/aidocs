import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn_string = os.getenv("SUPABASE_CONNECTION_STRING")
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

# Показать все колонки таблицы
cur.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'documents' AND table_schema = 'vecs'
    ORDER BY ordinal_position
""")

print("Columns in vecs.documents:")
for row in cur.fetchall():
    print(f"  {row[0]} ({row[1]})")

cur.close()
conn.close()
