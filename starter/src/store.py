"""Store and search vectors using PostgreSQL + pgvector."""
from __future__ import annotations

import json
from typing import Any

import psycopg

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """Retain connection settings; validate model dimensions."""
        self.database_url = database_url
        self.dimensions = int(dimensions)
        self.table_name = "strategy_chunks"

    def create_schema(self) -> None:
        """Create the vector table and useful indexes."""
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        strategy TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        chunk_text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        embedding VECTOR({self.dimensions}) NOT NULL
                    )
                    """
                )
                # CREATE TABLE IF NOT EXISTS does not change an existing
                # vector column when the configured embedding model changes.
                cursor.execute(
                    """
                    SELECT format_type(attribute.atttypid, attribute.atttypmod)
                    FROM pg_attribute AS attribute
                    JOIN pg_class AS relation ON relation.oid = attribute.attrelid
                    WHERE relation.relname = %s
                      AND attribute.attname = 'embedding'
                      AND NOT attribute.attisdropped
                    """,
                    (self.table_name,),
                )
                row = cursor.fetchone()
                expected_type = f"vector({self.dimensions})"
                if row and row[0].lower() != expected_type:
                    raise ValueError(
                        f"Embedding dimension mismatch: database column is {row[0]}, "
                        f"but the configured model produces {expected_type}. "
                        "Rebuild the index for the selected model before running ingestion."
                    )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_strategy_idx ON {self.table_name}(strategy)"
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_source_page_idx ON {self.table_name}(source, page)"
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx ON {self.table_name} USING GIN(metadata)"
                )
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table_name}_hnsw_idx ON {self.table_name} USING hnsw (embedding vector_cosine_ops)"
                )
                connection.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Validate lengths/dimensions and write stable IDs without duplicates."""
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts do not match")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(f"Expected {self.dimensions} dimensions, got {len(vector)}")

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {self.table_name} WHERE strategy = %s", (strategy,))
                for chunk, vector in zip(chunks, vectors):
                    chunk_id = f"{strategy}:{chunk.chunk_id}"
                    cursor.execute(
                        f"""
                        INSERT INTO {self.table_name} (id, strategy, document_id, source, page, chunk_text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        """,
                        (
                            chunk_id,
                            strategy,
                            chunk.document_id,
                            chunk.source,
                            chunk.page,
                            chunk.text,
                            json.dumps(chunk.metadata or {}),
                            list(map(float, vector)),
                        ),
                    )
                connection.commit()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter metadata in SQL, rank by cosine distance, and return top-k results."""
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Expected query vector dimension {self.dimensions}, got {len(query_vector)}")

        sql = (
            f"SELECT id AS chunk_id, chunk_text AS text, document_id, source, page, metadata, "
            f"1 - (embedding <=> %s::vector) AS score "
            f"FROM {self.table_name} WHERE strategy = %s"
        )
        params: list[Any] = [list(map(float, query_vector)), strategy]

        if filters:
            for key, value in filters.items():
                if key in {"source", "document_id"}:
                    sql += f" AND {key} = %s"
                    params.append(value)
                elif key == "page":
                    sql += " AND page = %s"
                    params.append(int(value))
                else:
                    sql += " AND metadata ->> %s = %s"
                    params.extend([str(key), str(value)])

        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([list(map(float, query_vector)), int(top_k)])

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            row_dict = {
                "chunk_id": row[0],
                "text": row[1],
                "document_id": row[2],
                "source": row[3],
                "page": row[4],
                "metadata": row[5] or {},
                "score": float(row[6]),
            }
            results.append(row_dict)
        return results
