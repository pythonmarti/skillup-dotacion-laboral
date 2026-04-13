"""Helpers para SQLite."""

import sqlite3
from contextlib import contextmanager

import pandas as pd
from config.settings import DB_PATH


@contextmanager
def get_connection(db_path=None):
    """Context manager para conexiones SQLite."""
    conn = sqlite3.connect(str(db_path or DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query_to_dataframe(query: str, params=None, db_path=None) -> pd.DataFrame:
    """Ejecuta un query y retorna un DataFrame."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)
