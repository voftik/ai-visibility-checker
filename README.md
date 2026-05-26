# AI Visibility Checker

Personal utility to probe websites as various AI crawlers (GPTBot, ClaudeBot,
PerplexityBot, etc.), capture how the site responds (status, redirects,
detected protections, robots.txt rules), and pass the aggregate to an LLM via
OpenRouter for a written analysis.

Single-user, runs on a single VPS. No auth, no Docker, no CI, no tests.

## Stack

- FastAPI + Uvicorn
- SQLite via SQLAlchemy 2.0 async (`aiosqlite`)
- httpx (async, HTTP/2) — used by the crawler
- Tailwind + Alpine.js via CDN; SSE for live logs
- LLM: OpenRouter, model `anthropic/claude-opus-4.7`

## Setup

```bash
uv sync
cp .env.example .env
# fill in OPENROUTER_API_KEY
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The SQLite file `sqlite.db` is created automatically at first start.

## Layout

```
app/
  main.py          FastAPI app + lifespan (init_db, event bus cleanup)
  config.py        pydantic-settings, reads .env
  db.py            async engine + session factory + init_db
  models.py        Run / DomainProbe / RobotsRule
  schemas.py       request/response Pydantic models
  routes/
    runs.py        CRUD for runs + SSE event stream
    pages.py       serves static/index.html
  services/
    crawler.py     async httpx probes + protections detection
    protections.py WAF / interstitial / SPA / TLS-block detectors
    robots_parser.py minimal RFC robots.txt parser
    analyzer.py    cross-probe pass + 4-step OpenRouter LLM pipeline
    event_bus.py   in-memory pub/sub for SSE
data/
  sets/                       JSON files served by /api/sets, surfaced in UI
scripts/
  extract_set2.py             rebuilds data/sets/set2_corpus_sources.json
static/index.html  Tailwind+Alpine skeleton
data/              source domain lists (gitignored content)
```

## Endpoints

- `GET /` — serves the SPA
- `POST /api/runs` — create a run, enqueue crawl in background
- `GET /api/runs` — last 50 runs
- `GET /api/runs/{id}` — full detail incl. probes and robots rules
- `DELETE /api/runs/{id}` — delete a run (cascades to probes/rules)
- `GET /api/runs/{id}/events` — SSE stream of log/progress events

## User-agents

The probe imitates the documented bot User-Agent strings of the major AI
providers: OpenAI (GPTBot, OAI-SearchBot, ChatGPT-User), Anthropic (ClaudeBot,
anthropic-ai, Claude-Web), Perplexity (PerplexityBot, Perplexity-User), plus
two control identities (Chrome browser baseline and an empty UA).

**DeepSeek (no official documentation).** DeepSeek is a Chinese AI provider
(DeepSeek-V3, DeepSeek-R1) without a public crawler spec, unlike OpenAI,
Anthropic and Perplexity. The two identities `DeepSeekBot` and `DeepSeek-User`
shipped here come from community block-lists (CrawlerCheck, Perishable Press,
knownagents.com, datadome.co) and from observations of how DeepSeek R1
performs two-stage retrieval; neither is confirmed by the vendor. Many sites
nevertheless target these names explicitly in `robots.txt` and WAF rules, so
probing them yields signal even without an official spec. Both checkboxes are
**off by default** in the UI — opt-in only, so a user understands the lower
fidelity of this imitation.

## SSE event shapes

```
{"type": "log",          "level": "...", "message": "...", "ts": "..."}
{"type": "progress",     "current": N, "total": M, "phase": "crawling|analyzing"}
{"type": "probe_done",   "domain": "...", "user_agent_label": "...", "http_status": ..., "summary": "..."}
{"type": "phase_change", "phase": "crawling_started|crawling_done|analyzing_started|analyzing_done|completed|failed"}
{"type": "final",        "status": "completed|failed"}
```

## Domain sets

Two pre-configured domain sets are exposed in the New run tab. Selection is
non-exclusive: both sets can be enabled, individual domains within a set can be
toggled, and custom domains can be added in the textarea below. The final list
is deduplicated client-side and sent to `POST /api/runs` together with a
`source_breakdown` block (`{set1_selected, set2_selected, custom,
deduplicated_removed}`) that is stored in `Run.config_json` and prepended to
the LLM analyzer's dataset text so the model knows where the sample came from.

**Set 1 — Manual research baseline.** 40 Russian domains from the original AI
visibility research, grouped by 12 categories, with annotations describing what
was observed via Claude `web_search`/`web_fetch` tools. Hardcoded in
`data/sets/set1_manual_research.json`. Edit the JSON to adjust the categories,
domains, or annotations. Annotations also surface as hover-tooltips on the
Probes Table in the Results section.

**Set 2 — Research corpus sources.** Domains extracted from the URL corpus
accumulated during AI visibility research (sources cited by LLM web search
agents). Sorted by citation frequency. Auto-extracted from
`domens_collection/all_sources.csv`.

### Updating Set 2

```bash
uv run python scripts/extract_set2.py
```

Run this manually after replacing `domens_collection/all_sources.csv`. The
script writes `data/sets/set2_corpus_sources.json` and prints a stats report
(URL count, filtered junk hosts, top 30 by frequency, category distribution).

### Adding a third set

Drop a JSON file into `data/sets/` following the structure of `set1` (object
with `categories: [{name, domains}]` and optional `annotations`) or `set2`
(object with `domains: [{domain, frequency, category_guess}]`). Reload the New
run tab; the card appears automatically.

## Production (systemd)

A unit at `/etc/systemd/system/ai-visibility.service` is already provisioned.
Enable once deps are installed:

```bash
systemctl enable --now ai-visibility
```

## Known limitations

- **`russian-tls-block` is vantage-point dependent.** The marker only fires
  when the TLS handshake actually fails, which depends on the trust store of
  the process. The Frankfurt VPS uses the Mozilla CA bundle, which cross-signs
  Минцифры roots, so domains like sberbank.ru complete the handshake from
  here even though they would fail on a clean Linux trust store. Reproducing
  the "blocked" state requires a separate trust-store path; we accept the
  false-negative on this vantage point rather than ship a custom CA store.
- **Single vantage point.** All probes go out from one IP/region. Geo blocks,
  ASN-based blocklists, and provider-specific peering effects are not
  measurable from a single VPS.
- **No JS execution.** The crawler does not render pages. SPA-only sites whose
  visible content is hydrated after load show up as `spa-empty-shell` — that
  is the correct signal for AI-crawler reachability (LLM crawlers also do not
  execute JS), but the body sample reflects the shell, not the rendered text.
