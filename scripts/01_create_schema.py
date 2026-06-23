"""Create the pgvector schema on Railway Postgres.

Runs migrations/001_init.sql. Requires DATABASE_URL with rights to CREATE EXTENSION;
on Railway's managed Postgres the default user has this.
"""
from __future__ import annotations

import pathlib
import sys

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.config import settings  # noqa: E402

SQL = (pathlib.Path(__file__).resolve().parents[1] / "migrations" / "001_init.sql").read_text()


def main() -> None:
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not set.")
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(SQL)
    print("Schema created / verified.")


if __name__ == "__main__":
    main()
