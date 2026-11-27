"""
Migration script to add detected_language column to evaluation_logs table.

Run this after updating the code to support multilingual language detection.

Usage:
    uv run python scripts/migrate_add_language_column.py
"""

import sqlite3
import logging
from graph_rag.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """Add detected_language column to evaluation_logs table if it doesn't exist."""
    db_path = settings.SQLITE_DB_PATH
    
    logger.info(f"Connecting to database: {db_path}")
    
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    
    try:
        # Check if column already exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(evaluation_logs)")
        columns = [row["name"] for row in cursor.fetchall()]
        
        if "detected_language" in columns:
            logger.info("✓ Column 'detected_language' already exists. No migration needed.")
            return
        
        # Add the column
        logger.info("Adding 'detected_language' column to evaluation_logs table...")
        cursor.execute(
            "ALTER TABLE evaluation_logs ADD COLUMN detected_language TEXT DEFAULT 'en'"
        )
        conn.commit()
        
        logger.info("✓ Migration completed successfully!")
        logger.info("  - Added 'detected_language' column with default value 'en'")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
