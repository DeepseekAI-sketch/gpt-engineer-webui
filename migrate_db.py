import sqlite3
import os
import json

# Path to your database file - update this path
DB_PATH = 'gpte.db'  # or the full path to your database

def migrate_database():
    print(f"Migrating database at {DB_PATH}...")
    
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if the column already exists
    cursor.execute("PRAGMA table_info(project)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    
    if 'project_metadata' not in column_names:
        print("Adding project_metadata column...")
        
        # Add the column - SQLite doesn't support JSON directly, so we use TEXT
        cursor.execute("ALTER TABLE project ADD COLUMN project_metadata TEXT")
        
        # If you previously had a 'metadata' column, migrate that data
        if 'metadata' in column_names:
            print("Migrating data from 'metadata' column...")
            cursor.execute("UPDATE project SET project_metadata = metadata")
        
        conn.commit()
        print("Migration successful!")
    else:
        print("Column project_metadata already exists. No migration needed.")
    
    conn.close()

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"Database file {DB_PATH} not found!")
    else:
        migrate_database()