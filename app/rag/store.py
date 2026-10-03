"""Persist RAG chunks and embeddings into Supabase Postgres (pgvector).

Kept in a module (not the notebook) so the adapter logic is stable and testable.
Uses no scikit-learn, so it imports fine under Windows Smart App Control.
"""

from __future__ import annotations

import os

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

load_dotenv(override=True)


def store_chunk(content: str, metadata: dict, embedding) -> None:
    with psycopg.connect(os.getenv("SUPABASE_DB_URL")) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                (content, metadata, embedding)
                VALUES (%s, %s, %s)
                """,
                (
                    content,
                    Jsonb(metadata),
                    np.asarray(embedding, dtype=np.float32),
                ),
            )
        conn.commit()
