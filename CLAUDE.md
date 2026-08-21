# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` is the long-form developer guide (Russian) and is current — read it for the
full pipeline, metric definitions and API contracts. This file is the short orientation
plus the traps that are not obvious from the code.

## Project

RW+ AI Visibility Checker (AIV) — the user submits **one domain** and gets an executive
report on how visible that site and brand are to AI answer engines. One run does:

1. crawls the site as several AI/search crawlers (GPTBot, ClaudeBot, PerplexityBot,
   Googlebot family, …), recording HTTP status, redirects, WAF/anti-bot protections and
   robots.txt rules, plus a Chromium screenshot of the real first screen;
2. builds a site profile, then researches the market with attested web search;
3. designs nine INTENT scenarios and asks a panel of five model providers — once with
   web search, once from memory;
4. resolves entities in the answers, annotates them, computes deterministic metrics,
   runs a bounded critic loop;
5. writes the final report, passes it through a semantic gate, and generates
   illustrations.

Single-user tool on one VPS; no auth, no Docker, no CI. UI text is Russian.

## Commands

```bash
uv sync                                                            # install deps (Python >=3.12)
cp .env.example .env                                               # then set OPENROUTER_API_KEY
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload    # dev server

uv run python -m unittest discover -s tests                        # all tests (~400)
uv run python -m unittest tests.test_openrouter_policy -v          # one module
```

No linter or formatter is configured (no ruff/pytest — tests are stdlib `unittest`). Run
tests from the repo root: they use relative paths like `static/index.html`.

`tests/test_frontend_*_runtime.py` shell out to `node` to execute the SPA's JavaScript.
Node is **not** installed on the VPS, so those two modules error there and pass locally.
Everything else must be green in both places.

**Tests touch the real `sqlite.db`.** Several suites insert and delete rows through the
production `SessionLocal` (cleanup in `finally`). Back the file up before running them
against production data.

**Local working copy vs VPS.** This directory (under `Yandex.Disk.localized`) mirrors the
deployment at `webtest:/root/projects/ai-visibility-checker`, which has **no git**. The
mirror is Yandex.Disk-synced across machines, so conflicting edits show up as `… (2).py`
copies — check for them before assuming a file is the only version. Deploy is an explicit
rsync; nothing propagates on its own:

```bash
rsync -a --exclude='.git/' --exclude='.venv/' --exclude='backups/' --exclude='__pycache__/' \
      --exclude='static/generated/' --exclude='sqlite*.db*' --exclude='.env' \
      --exclude='.proxy_cache.json' --exclude='output/' ./ webtest:/root/projects/ai-visibility-checker/
ssh webtest systemctl restart ai-visibility
```

Branch `legacy/multi-domain-crawler` holds the abandoned earlier product (multi-domain
crawler with domain sets). Nothing on `main` should reference it.

## Architecture

FastAPI + SQLAlchemy 2.0 async (aiosqlite) + a single-file vanilla-JS SPA. Everything runs
in one process; the run coordinator, the SSE event bus and the proxy pool are in-process
state, so **the app must run as a single uvicorn worker** (never add `--workers N`).

### Run lifecycle

1. `POST /api/runs {"domain": …}` (`app/routes/runs.py`) normalizes the domain, reuses an
   already-active run for the same domain if there is one, and inserts a `pending` row.
   Nothing is started inline.
2. `app/services/run_coordinator.py` owns a single durable execution slot
   (`EXECUTION_SLOT = 1`) in SQLite. It claims one `pending` run at a time under a lease
   (`RUN_LEASE_SECONDS`, heartbeat), so exactly one check runs at a time and an abandoned
   lease is recovered on the next poll. `coordinator.wake()` shortcuts the poll interval.
3. `app/services/crawler.py` discovers representative pages, fans out
   `pages × user agents` probes under a semaphore, and writes `DomainProbe`, `RobotsRule`
   and `SitePage` rows. `USER_AGENT_STRINGS` / `USER_AGENT_PROFILES` are the canonical
   bot-identity registry — UI labels, analyzer constants and tests all key off them.
4. `app/services/analyzer.py::analyze_run` then walks the stages below, persisting every
   LLM step as a `RunArtifact`. On any exception it calls `fail_run()` with a message
   inviting the user to press «Продолжить»; `POST /api/runs/{id}/retry` puts the run back
   to `pending` and the coordinator replays it, reusing saved artifacts.

Stage keys (`app/services/progress.py::STAGES`) and their percent bands:

| stage_key | label | band |
|---|---|---|
| `site_discovery` | Изучаем сайт | 0–17 |
| `technical_access` | Проверяем доступ для ИИ | 17–28 |
| `scenario_design` | Формируем сценарии выбора | 28–38 |
| `web_visibility` | Сравниваем ответы ИИ-систем | 38–64 |
| `knowledge_gap` | Считаем видимость и разрывы знаний | 65–82 |
| `report` | Собираем отчёт и иллюстрации | 82–100 |

### Web-access contract (the part that bites)

Every OpenRouter call declares a `WebSearchPolicy` — `REQUIRED`, `NATIVE_REQUIRED`
(Perplexity Sonar) or `FORBIDDEN`. `web_request_policy()` turns that into request fields,
and `attest_web_response()` verifies **after the fact**, from
`usage.server_tool_use_details.web_search_requests` and `url_citation` annotations, that
retrieval really happened. A failed attestation raises `OpenRouterPolicyError`, which
`chat()` retries — the retry now carries an explicit note about what was rejected.

The attestation is the guarantee. Do **not** try to force retrieval on the request side:
sending `tool_choice: "required"` alongside the `openrouter:web_search` server tool holds
the constraint on every turn of the tool loop, so the model never gets a turn to write an
answer — Anthropic ends it with `native_finish_reason="pause_turn"` and empty content.
That silently broke every run between 2026-08-01 and 2026-08-21.

**Citations are endpoint-dependent, and OpenRouter routes by default.** The same model can
run the search and still return zero `url_citation` annotations, depending on which
upstream endpoint answered. Measured 2026-08-21 on `anthropic/claude-opus-5`, identical
request, provider pinned:

| endpoint | annotations | web_search_requests |
|---|---:|---:|
| Anthropic | 0 | 3 |
| Claude Platform on AWS | 0 | 3 |
| Amazon Bedrock | 11–12 | 4 |
| Azure | 11 | 4 |

Reasoning settings make no difference. Since default routing picks Claude Platform on AWS,
`market_research` could never satisfy its gate. `chat()` therefore remembers, per model,
endpoints that searched but returned no citations, and routes around them with
`provider.ignore` — in-process memory, reset on restart, no hardcoded provider table. That
routing stays **out of** `request_policy`: the contract records what was required, not
which endpoint proved it, and folding it in would change the policy hash and invalidate
already-collected panel cells.

Consequence when debugging: a first attempt landing on a non-citing endpoint is normal and
costs one call; the retry is where it succeeds.

### Other subsystems

- `app/services/protections.py` — WAF / interstitial / SPA-empty-shell / TLS-block
  detectors; `content_extractor.py` — text-extractability signals stored on probes.
- `app/services/site_preview.py` + `site_preview_worker.py` — Chromium screenshot of the
  first screen; in production an isolated worker via `SITE_PREVIEW_WORKER_COMMAND`.
- `app/services/proxy_pool.py` — optional Webshare.io outbound proxy pool, initialized in
  the `lifespan` hook (`app/main.py`). Empty `WEBSHARE_API_KEY` = direct mode; per-probe
  direct fallback on proxy errors (`PROXY_FALLBACK_DIRECT`).
- `app/services/event_bus.py` — in-memory pub/sub feeding SSE `GET /api/runs/{id}/events`,
  which replays missed events via `Last-Event-ID`.
- Sharing: `POST /api/runs/{id}/share` mints a `secrets.token_urlsafe` token on
  `Run.share_token`; public read-only access via `GET /api/shared/{token}`
  (`app/routes/shared.py`) and SPA route `GET /r/{token}` (`app/routes/pages.py`).

### Data

Tables: `runs`, `domain_probes`, `robots_rules`, `site_pages`, `run_artifacts`,
`visibility_prompts`, `model_answers`, `answer_annotations`, `report_illustrations`.

`run_artifacts` is the resume backbone: one row per LLM step, keyed by
`(run_id, stage_key, artifact_key)` with `status`, `input_json`, `output_json`, `raw_text`,
`usage_json`, `error_message` and a `prompt_version`. A retry reuses `completed` artifacts
whose input still matches, so bumping a `*_VERSION` constant deliberately invalidates the
cache. When a run fails, look here first — `error_message` names the step.

No Alembic. `init_db()` in `app/db.py` runs `create_all` plus idempotent `ALTER TABLE`
statements (`_ALTER_STATEMENTS`) for pre-existing DBs — add new columns there. It also
enables SQLite foreign keys per connection and, on startup, marks any run stuck in
`pending/crawling/analyzing` as failed (its task died with the process).

### Frontend

The entire UI is `static/index.html` — vanilla HTML/CSS/JS with vendored assets in
`static/vendor/` (no Node toolchain in the repo). Dark RW+ branding via `--brand-*` CSS
custom properties; palette/logo notes in `static/brand/BRAND_NOTES.md`. Per-run generated
images live in `static/generated/<run-id>/` and are gitignored.
`tests/test_frontend_static.py` asserts on raw HTML/CSS strings, so UI copy or CSS changes
can break it — update the test alongside.

### Config

pydantic-settings reading `.env` (`app/config.py`). Model selection is split across
several variables, and **the analysis pipeline reads `OPENROUTER_ANALYSIS_MODEL`, not
`OPENROUTER_MODEL`** — setting only the latter in `.env` silently changes nothing for the
report. The panel providers have their own variables
(`OPENROUTER_OPENAI_MODEL`, `_GEMINI_`, `_PERPLEXITY_`, `_DEEPSEEK_`, `_CLAUDE_`).

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

Runs as systemd unit `ai-visibility.service` behind nginx at `aiv.rw.plus` (uvicorn on
127.0.0.1:8000, `proxy_buffering off` and `proxy_read_timeout 1800s` for SSE). Deploy is
the rsync above followed by `systemctl restart ai-visibility`; there is no git on the VPS.

The unit caps memory (`MemoryHigh=900M`, `MemoryMax=1200M`) on a 1.9 GiB box and allows
90 s for shutdown so the coordinator can checkpoint an in-flight run.

Diagnosing a stuck or failed run, in order: `runs.error_message` and `stage_key`, then the
matching `run_artifacts` row, then `journalctl -u ai-visibility` for the traceback. The
user-facing message is deliberately generic — it never names the failing step.

README's «Известные ограничения» section explains vantage-point caveats
(russian-tls-block detection, single IP, no JS rendering).
