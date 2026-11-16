# Repository Guidelines

## Project Structure

This repository is a Python data-ingestion lab. Core assignment guidance and evaluation criteria live in the root Markdown files (`GUIDE.md`, `BENCHMARK.md`, `RULES.md`, and `SUBMISSION.md`). The optional PostgreSQL + pgvector starter is under `starter/`: pipeline modules are in `starter/src/`, setup checks in `starter/scripts/`, sample corpus and questions in `starter/data/` and `starter/benchmark/`, and generated summaries in `starter/results/`. Supporting diagrams are in `assets/`; student reports are in `report/individual/`. There is no dedicated automated test suite currently.

## Build, Run, and Development

Run commands from `starter/` unless noted:

- `python3 -m venv .venv` and `python -m pip install -r requirements.txt` create an environment and install core dependencies.
- `docker compose -p day6_student_starter up -d --wait` starts the local pgvector database.
- `python scripts/check_setup.py` verifies database connectivity and the pgvector extension; it does not run the pipeline.
- `python -m src.run` ingests the corpus, embeds chunks, evaluates retrieval strategies, and writes `results/summary.json`.

Copy `.env.example` to `.env` and configure the database and embedding model before running. Install `sentence-transformers` separately for local Sentence Transformers models. Keep model and dimensions consistent between document and query embeddings; rebuild the index when changing models.

## Coding Style

Use Python 3.11+, four spaces for indentation, `snake_case` for functions and variables, and `CapWords` for classes. Keep pipeline stages in their focused `starter/src/` modules and use type hints and concise docstrings, following existing code. No formatter or linter is configured; keep changes readable and avoid unrelated generated outputs.

## Verification

No pytest suite or coverage requirement is configured. For changes affecting setup, run `python scripts/check_setup.py`; for an end-to-end change, run `python -m src.run` and inspect the summary and embedding metadata. Do not claim benchmark results unless the pipeline actually completed. Keep benchmark questions and evidence in `starter/benchmark/questions.json` consistent with the root `BENCHMARK.md` requirements.

## Commits and Pull Requests

Recent commits use short imperative summaries, often with a type prefix, such as `feat: ...` and `Remove ...`. Follow that pattern and keep each commit focused. A pull request should explain its purpose, list affected pipeline/configuration files, include the exact verification commands and outcomes, and note any changes to corpus, benchmark, embedding model, dimensions, or generated results. Link related issues when applicable.

## Security and Data

Never commit `.env`, API keys, personal or confidential data, virtual environments, or logs containing secrets. Use only data you are permitted to process and distribute; record its source and license or permission in the manifest/report. Do not silently replace failed real embeddings with mock or random vectors.
