import lancedb
import json
import pandas as pd
import os

db_uri = "rust/src-tauri/data/lancedb/data/lancedb"

def inspect_table(db, table_name):
    print(f"\n{'='*20} Table: {table_name} {'='*20}")
    try:
        tbl = db.open_table(table_name)
        df = tbl.to_pandas().head(10)
        if df.empty:
            print(f"Table '{table_name}' is empty.")
            return
        print(f"Columns: {df.columns.tolist()}")
        for i, row in df.iterrows():
            print(f"\n--- Row {i+1} ---")
            d = row.to_dict()
            if 'vector' in d: d['vector'] = "VECTOR_DATA"
            if 'data' in d and isinstance(d['data'], str):
                try: d['data'] = json.loads(d['data'])
                except: pass
            print(json.dumps(d, indent=2, ensure_ascii=False, default=str))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if os.path.exists(db_uri):
        db = lancedb.connect(db_uri)
        for name in ["pages", "users", "items", "tasks"]:
            inspect_table(db, name)
    else:
        print(f"Path not found: {db_uri}")