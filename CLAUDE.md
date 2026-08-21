# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RW+ AI Visibility Checker — probes websites as various AI crawlers (GPTBot, ClaudeBot, PerplexityBot, Googlebot family, …), records how each site responds (HTTP status, redirects, detected WAF/anti-bot protections, robots.txt rules), and passes the aggregate to an LLM via OpenRouter for a written analysis. Single-user tool on one VPS; no auth, no Docker, no CI. UI text is Russian.

## Commands

```bash
uv sync                                                            # install deps (Python >=3.12)
cp .env.example .env                                               # then set OPENROUTER_API_KEY
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload    # dev server

uv run python -m unittest discover -s tests -v                     # all tests
uv run python -m unittest tests.test_pipeline_safety -v            # one module
uv run python -m unittest tests.test_pipeline_safety.RobotsParserTests.test_specific_bot_group_overrides_wildcard  # one test

uv run python scripts/extract_set2.py    # rebuild data/sets/set2_corpus_sources.json from domens_collection/all_sources.csv
```

No linter or formatter is configured (no ruff/pytest — tests are stdlib `unittest`). Run tests from the repo root: they use relative paths like `static/index.html`.

**Local working copy vs VPS.** This directory (under `Yandex.Disk.localized`) is a synced mirror of the production VPS deployment; `uv` and the dependency environment exist on the VPS, not necessarily on the local Mac (no `.venv` here, system `python3` lacks fastapi/httpx). Locally, only the stdlib-only module runs: `python3 -m unittest tests.test_frontend_static -v`. The other test modules need the deps installed.

**Tests touch the real `sqlite.db`.** `DatabaseSafetyTests` in `tests/test_pipeline_safety.py` inserts and deletes rows through the production `SessionLocal` (cleanup in `finally`). Don't run them against a database whose contents you can't risk.

## Architecture

FastAPI + SQLAlchemy 2.0 async (aiosqlite) + a single-file Alpine.js SPA. Everything runs in one process; background crawl tasks, the SSE event bus, and the proxy pool are all in-process state, so **the app must run as a single uvicorn worker** (never add `--workers N`).

Run lifecycle (the core flow):
1. `POST /api/runs` (`app/routes/runs.py`) validates config, creates a `Run` row, and starts `run_crawl(run_id)` via `asyncio.create_task`. Tasks are kept in module-level `_run_tasks` so `DELETE /api/runs/{id}` can cancel an in-flight crawl before the cascade delete fires.
2. `app/services/crawler.py` fans out `domains × (selected UAs + robots-fetcher)` probe jobs under an `asyncio.Semaphore`, writing one `DomainProbe` row per probe and `RobotsRule` rows from parsed robots.txt. Its `USER_AGENT_STRINGS` / `USER_AGENT_PROFILES` dicts are the canonical bot-identity registry — UI labels, analyzer constants, and tests all key off these labels.
3. `app/services/analyzer.py` runs after the crawl in the same task: a deterministic cross-probe pass (`apply_ua_conditional_block`: AI bot blocked while Chrome-control succeeds), a dataset-text builder with char budgets (compact mode past ~100K chars), then 4 sequential OpenRouter LLM calls (steps 2–5). Final report → `Run.analysis_markdown`; intermediates → `Run.config_json["intermediate_analysis"]`. Any failure → `status=failed`, raw probe data kept.
4. Progress/log events flow through `app/services/event_bus.py` (in-memory pub/sub with per-channel history) to SSE endpoint `GET /api/runs/{id}/events`, which replays missed events via `Last-Event-ID`. Event shapes are documented in README.md.

Other subsystems:
- `app/services/protections.py` — WAF / interstitial / SPA-empty-shell / TLS-block detectors; `content_extractor.py` — text-extractability signals stored on probes.
- `app/services/proxy_pool.py` — optional Webshare.io outbound proxy pool, initialized in the `lifespan` hook (`app/main.py`). Empty `WEBSHARE_API_KEY` = direct mode; per-probe direct fallback on proxy errors (`PROXY_FALLBACK_DIRECT`).
- Sharing: `POST /api/runs/{id}/share` mints a `secrets.token_urlsafe` token on `Run.share_token`; public read-only access via `GET /api/shared/{token}` (`app/routes/shared.py`) and SPA route `GET /r/{token}` (`app/routes/pages.py` serves the same `index.html`; Alpine reads the token from `location.pathname`).
- Domain sets: JSON files dropped into `data/sets/` are auto-discovered by `app/routes/sets.py` and appear as cards in the UI. Sets 1/3 use `{categories: [{name, domains}], annotations}`; set 2 uses `{domains: [{domain, frequency, category_guess}]}`.

Database / migrations: no Alembic. `init_db()` in `app/db.py` runs `create_all` plus idempotent `ALTER TABLE` statements (`_ALTER_STATEMENTS`) for pre-existing DBs — add new columns there. It also enables SQLite foreign keys per connection and, on startup, marks any run stuck in `pending/crawling/analyzing` as failed (its asyncio task died with the process).

Frontend: the entire UI is `static/index.html` (~4300 lines) — Alpine.js with vendored assets in `static/vendor/` (prebuilt minified Tailwind CSS, `alpine.min.js`, `marked.min.js`; no Node toolchain in the repo). Dark RW+ branding via `--brand-*` CSS custom properties; palette/logo notes in `static/brand/BRAND_NOTES.md`. Note: `tests/test_frontend_static.py` asserts on raw HTML/CSS strings, so UI copy or CSS changes can break it — update the test alongside.

Config: pydantic-settings reading `.env` (`app/config.py`). LLM model selected by `OPENROUTER_MODEL`.

## Reprocessing saved answers

Rebuilding annotations, metrics and the report from answers that are already in the
database is an **operator procedure with one supported entrypoint**:

```bash
systemctl start aiv-reprocess@<run_id>    # template: scripts/aiv-reprocess@.service
journalctl -fu aiv-reprocess@<run_id>     # follow it
systemctl stop aiv-reprocess@<run_id>     # safe: the run state is rolled back
```

Without systemd, the same thing: `uv run python scripts/reprocess_saved_run.py <run_id>`.

**Never call `analyzer.reprocess_saved_answers()` directly** — not through `python -c`,
not through `systemd-run`. That skips everything `scripts/reprocess_saved_run.py` does:
claiming the run, holding the durable execution lease with a 30 s heartbeat,
fingerprinting the raw answer corpus with SHA-256, snapshotting `PreviousRunState` and
restoring it if the rebuild fails.

The bypass fails silently rather than loudly: `assert_run_lease()` is a no-op when no
lease is bound, because `lease_owner_for()` returns `None` (`app/services/run_lease.py`).
An unowned writer then races the coordinator and, if it dies, leaves the run stuck in
`analyzing` until the lease expires. This is exactly what happened on 2026-07-30: a
`systemd-run` one-liner was killed after 2 min and left a failed transient unit sitting
in systemd for two and a half weeks.

Reprocessing spends tokens on the analysis and report layers, but never re-queries the
model panel — that is what "saved answers" means.

Both systemd units are kept in `scripts/` and mirrored into `/etc/systemd/system/`.
After editing either one:

```bash
install -m 0644 scripts/ai-visibility.service /etc/systemd/system/
install -m 0644 'scripts/aiv-reprocess@.service' /etc/systemd/system/
systemctl daemon-reload
```

The main unit sets `SuccessExitStatus=143`: `uv run` exits 128+15 on SIGTERM, so without
it every ordinary `systemctl stop` would mark the service failed.

## Production

Runs as systemd unit `ai-visibility.service` behind nginx at `aiv.rw.plus`; restart with `systemctl restart ai-visibility` after deploy. README's "Known limitations" section explains vantage-point caveats (russian-tls-block detection, single IP, no JS rendering).
