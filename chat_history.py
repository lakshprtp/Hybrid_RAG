import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ChatHistory")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SQLITE_DB_PATH = os.getenv("CHAT_HISTORY_DB_PATH", "./chat_history.db")

# Flag indicating whether we are using PostgreSQL
USE_POSTGRES = bool(DATABASE_URL and (DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")))

if USE_POSTGRES:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        logger.info("Configured for PostgreSQL storage.")
    except ImportError:
        logger.warning("psycopg2-binary not found, falling back to SQLite.")
        USE_POSTGRES = False
        import sqlite3
else:
    import sqlite3


# ==========================================
# Connection Providers
# ==========================================

def get_postgres_connection():
    """Create a new PostgreSQL connection with dictionary cursor."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    url = DATABASE_URL
    # Render sometimes provides postgres:// which psycopg2 accepts, but normalizes cleanly
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    # Enable SSL mode if not specified (needed for Render / Supabase / Neon)
    if "sslmode=" not in url and ("render.com" in url or "supabase.co" in url or "neon.tech" in url):
        connector = "&" if "?" in url else "?"
        url = f"{url}{connector}sslmode=require"
        
    return psycopg2.connect(url, cursor_factory=RealDictCursor)


def get_sqlite_connection():
    """Create a new SQLite connection with dictionary row factory."""
    import sqlite3
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ==========================================
# Schema Initialization
# ==========================================

def init_db():
    """Create tables if they do not exist."""
    global USE_POSTGRES
    if USE_POSTGRES:
        try:
            with get_postgres_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS threads (
                            id VARCHAR(64) PRIMARY KEY,
                            title VARCHAR(255) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS messages (
                            id VARCHAR(64) PRIMARY KEY,
                            thread_id VARCHAR(64) NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                            role VARCHAR(32) NOT NULL,
                            content TEXT NOT NULL,
                            doc_sources TEXT DEFAULT '[]',
                            web_sources TEXT DEFAULT '[]',
                            stats TEXT DEFAULT '{}',
                            mode_used VARCHAR(32) DEFAULT '',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    conn.commit()
            logger.info("PostgreSQL chat history schema initialized successfully.")
            return
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}. Falling back to SQLite.")
            USE_POSTGRES = False

    # SQLite fallback
    with get_sqlite_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                doc_sources TEXT DEFAULT '[]',
                web_sources TEXT DEFAULT '[]',
                stats TEXT DEFAULT '{}',
                mode_used TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
    logger.info("SQLite chat history schema initialized successfully.")


# Initialize on import
init_db()


def get_storage_engine() -> str:
    """Return the name of the active database engine."""
    return "Postgres"


# ==========================================
# CRUD Operations
# ==========================================

def create_thread(title: str = "New Chat") -> str:
    """Create a new chat thread and return its ID."""
    thread_id = str(uuid.uuid4())
    now = datetime.now()

    if USE_POSTGRES:
        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO threads (id, title, created_at, updated_at) VALUES (%s, %s, %s, %s);",
                    (thread_id, title, now, now)
                )
                conn.commit()
    else:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
                (thread_id, title, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()

    return thread_id


def list_threads() -> List[Dict[str, Any]]:
    """List all threads ordered by latest activity."""
    if USE_POSTGRES:
        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM threads ORDER BY updated_at DESC;")
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
    else:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM threads ORDER BY updated_at DESC;")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


def get_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific thread."""
    if USE_POSTGRES:
        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM threads WHERE id = %s;", (thread_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
    else:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM threads WHERE id = ?;", (thread_id,))
            row = cursor.fetchone()
            return dict(row) if row else None


def update_thread_title(thread_id: str, new_title: str):
    """Update title and timestamp of a thread."""
    now = datetime.now()
    if USE_POSTGRES:
        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE threads SET title = %s, updated_at = %s WHERE id = %s;",
                    (new_title, now, thread_id)
                )
                conn.commit()
    else:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?;",
                (new_title, now.strftime("%Y-%m-%d %H:%M:%S"), thread_id)
            )
            conn.commit()


def delete_thread(thread_id: str):
    """Delete a thread and its messages."""
    if USE_POSTGRES:
        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM messages WHERE thread_id = %s;", (thread_id,))
                cursor.execute("DELETE FROM threads WHERE id = %s;", (thread_id,))
                conn.commit()
    else:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE thread_id = ?;", (thread_id,))
            cursor.execute("DELETE FROM threads WHERE id = ?;", (thread_id,))
            conn.commit()


def add_message(
    thread_id: str,
    role: str,
    content: str,
    doc_sources: Optional[List[Dict[str, Any]]] = None,
    web_sources: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[Dict[str, Any]] = None,
    mode_used: str = ""
) -> str:
    """Save a new chat message to a thread."""
    msg_id = str(uuid.uuid4())
    now = datetime.now()
    doc_sources_json = json.dumps(doc_sources or [])
    web_sources_json = json.dumps(web_sources or [])
    stats_json = json.dumps(stats or {})

    if USE_POSTGRES:
        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO messages (id, thread_id, role, content, doc_sources, web_sources, stats, mode_used, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (msg_id, thread_id, role, content, doc_sources_json, web_sources_json, stats_json, mode_used, now)
                )
                cursor.execute("UPDATE threads SET updated_at = %s WHERE id = %s;", (now, thread_id))
                conn.commit()
    else:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (id, thread_id, role, content, doc_sources, web_sources, stats, mode_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (msg_id, thread_id, role, content, doc_sources_json, web_sources_json, stats_json, mode_used, now.strftime("%Y-%m-%d %H:%M:%S"))
            )
            cursor.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ?;",
                (now.strftime("%Y-%m-%d %H:%M:%S"), thread_id)
            )
            conn.commit()

    return msg_id


def get_thread_messages(thread_id: str) -> List[Dict[str, Any]]:
    """Retrieve all messages for a specific thread."""
    if USE_POSTGRES:
        with get_postgres_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM messages WHERE thread_id = %s ORDER BY created_at ASC;",
                    (thread_id,)
                )
                rows = cursor.fetchall()
                messages = [dict(r) for r in rows]
    else:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM messages WHERE thread_id = ? ORDER BY created_at ASC;",
                (thread_id,)
            )
            rows = cursor.fetchall()
            messages = [dict(r) for r in rows]

    # Parse JSON fields
    for msg in messages:
        if isinstance(msg.get("doc_sources"), str):
            try:
                msg["doc_sources"] = json.loads(msg["doc_sources"])
            except Exception:
                msg["doc_sources"] = []
        if isinstance(msg.get("web_sources"), str):
            try:
                msg["web_sources"] = json.loads(msg["web_sources"])
            except Exception:
                msg["web_sources"] = []
        if isinstance(msg.get("stats"), str):
            try:
                msg["stats"] = json.loads(msg["stats"])
            except Exception:
                msg["stats"] = {}

    return messages
