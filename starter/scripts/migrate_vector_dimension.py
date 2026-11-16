"""Archive an incompatible vector table without dropping its rows."""
from __future__ import annotations

import os
import re

import psycopg
from psycopg import sql
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is missing; configure starter/.env first.")

    dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    if dimensions < 1:
        raise SystemExit("EMBEDDING_DIMENSIONS must be a positive integer.")

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.strategy_chunks')")
            if cursor.fetchone()[0] is None:
                print("strategy_chunks does not exist; nothing to migrate.")
                return

            cursor.execute(
                """
                SELECT format_type(attribute.atttypid, attribute.atttypmod)
                FROM pg_attribute AS attribute
                WHERE attribute.attrelid = 'public.strategy_chunks'::regclass
                  AND attribute.attname = 'embedding'
                  AND NOT attribute.attisdropped
                """
            )
            row = cursor.fetchone()
            if row is None:
                raise SystemExit("strategy_chunks.embedding was not found; refusing migration.")
            current_type = row[0].lower()
            expected_type = f"vector({dimensions})"
            if current_type == expected_type:
                print(f"Already configured for {expected_type}; nothing to migrate.")
                return

            suffix = re.sub(r"[^a-zA-Z0-9_]", "_", current_type)
            archive_table = f"strategy_chunks_archive_{suffix}"
            cursor.execute("SELECT to_regclass(%s)", (f"public.{archive_table}",))
            if cursor.fetchone()[0] is not None:
                raise SystemExit(
                    f"Archive table {archive_table} already exists; refusing to overwrite it. "
                    "Rename or inspect it manually before retrying."
                )

            # Keep the old table and all its rows intact. Rename attached
            # indexes first so create_schema() can create fresh index names.
            cursor.execute(
                """
                SELECT index_class.relname
                FROM pg_index AS index_info
                JOIN pg_class AS table_class ON table_class.oid = index_info.indrelid
                JOIN pg_namespace AS table_ns ON table_ns.oid = table_class.relnamespace
                JOIN pg_class AS index_class ON index_class.oid = index_info.indexrelid
                WHERE table_ns.nspname = 'public' AND table_class.relname = 'strategy_chunks'
                """
            )
            index_names = [item[0] for item in cursor.fetchall()]
            for index_name in index_names:
                cursor.execute(
                    sql.SQL("ALTER INDEX public.{} RENAME TO {}").format(
                        sql.Identifier(index_name), sql.Identifier(f"{index_name}_archived")
                    )
                )
            cursor.execute(
                sql.SQL("ALTER TABLE public.strategy_chunks RENAME TO {}").format(
                    sql.Identifier(archive_table)
                )
            )
            connection.commit()

    print(f"Archived existing {current_type} rows as public.{archive_table}.")
    print(f"The ingestion run will create a fresh {expected_type} table and re-embed all strategies.")


if __name__ == "__main__":
    main()
