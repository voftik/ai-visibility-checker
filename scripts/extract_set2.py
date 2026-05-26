"""Extract domains from domens_collection/all_sources.csv into data/sets/set2_corpus_sources.json.

CSV format observed: header `url;model`, semicolon-separated, CRLF, ASCII.
Frequency = number of times a domain appears across all rows (NOT deduped per model).

Usage:
    uv run python scripts/extract_set2.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "domens_collection" / "all_sources.csv"
OUT_PATH = ROOT / "data" / "sets" / "set2_corpus_sources.json"
OVERRIDES_PATH = ROOT / "data" / "set2_category_overrides.json"


def _load_overrides() -> dict[str, str]:
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        raw = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, str) and not k.startswith("_")}

# Drop hosts that appear fewer than MIN_FREQUENCY times in the corpus.
# Long-tail domains with one or two citations rarely add signal to the LLM
# analysis — they inflate the dataset without changing the structural picture.
MIN_FREQUENCY = 10

# Strip these substrings or exact-matches from the result. Each entry is
# either an exact host match or a host suffix (".example.com" matches
# subdomains too).
JUNK_DOMAINS_EXACT = {
    "archive.org",
    "web.archive.org",
    "scholar.google.com",
    "books.google.com",
    "translate.google.com",
    "docs.google.com",
    "drive.google.com",
    "youtube.com",
    "youtu.be",
    "m.youtube.com",
    "www.youtube.com",
    "bit.ly",
    "t.co",
    "tinyurl.com",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "localhost",
    "127.0.0.1",
    "example.com",
    "example.org",
}


def normalize_host(netloc: str) -> str:
    h = netloc.strip().lower()
    if not h:
        return ""
    # strip user:pass@
    if "@" in h:
        h = h.split("@", 1)[1]
    # strip :port
    if ":" in h:
        h = h.split(":", 1)[0]
    if h.startswith("www."):
        h = h[4:]
    return h


ALLOWED_TLDS = (".ru", ".рф", ".xn--p1ai", ".su")


def is_junk(host: str) -> bool:
    if host in JUNK_DOMAINS_EXACT:
        return True
    # bare IP literals
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        return True
    return False


def has_allowed_tld(host: str) -> bool:
    return host.endswith(ALLOWED_TLDS)


def category_guess(host: str) -> str:
    """Best-effort categorisation. Falls back to 'other' so every domain has a tag.

    Order matters: first match wins. Russian-TLD-focused heuristics.
    """
    h = host.lower()

    # Yandex ecosystem (must come before generic categories since some yandex
    # subdomains contain words like "market" or "news").
    if h == "yandex.ru" or h.endswith(".yandex.ru") or h == "ya.ru" or h.endswith(".ya.ru"):
        return "yandex"
    # Mail.ru / VK / OK family
    if any(k in h for k in ("mail.ru", "vk.com", "vk.ru", "ok.ru")):
        return "mailru_vk"
    # Telegram
    if h == "t.me" or h.endswith(".t.me"):
        return "telegram"
    # Government
    if h.endswith(".gov.ru") or h.endswith(".gov") or h.endswith(".gov.uk") or h.endswith(".gov.us"):
        return "government"
    if any(k in h for k in ("kremlin", "duma", "mvd", "gosuslugi", "mos.ru", "nalog", "cbr.ru", "minfin")):
        return "government"
    # Finance / banking / personal finance / investments
    if any(k in h for k in (
        "banki", "bankir", "bankprofi", "sravni", "mainfin", "vsezaim",
        "1000bankov", "vbr", "finuslugi", "kredit", "zaim", "rocketbank",
        "tinkoff", "alfa", "sber", "vtb", "gazprombank", "tbank", "t-j.",
        "hranideng", "vse-dengy", "smart-lab", "xvestor", "brobank",
        "finforum", "finance", "fin.", "invest", "broker", "capital",
        "money", "moneydom", "incom",
    )):
        return "finance"
    # Adtech / digital marketing services / SEO agencies
    if any(k in h for k in (
        "elama", "adpass", "advertising", "akarussia", "demis", "ingate",
        "calltouch", "completo", "andata", "click.ru", "callibri",
        "sales-generator", "seojazz", "digitalstrategy", "realweb",
        "cmsmagazine", "yagla", "marketolog", "yak-studio", "agency-perform",
        "rush-agency", "iconext", "k50.ru", "calltracking", "leadgen",
        "advmaker", "performance-",
    )):
        return "adtech"
    # Marketing / business / startup media (Russian)
    if any(k in h for k in (
        "sostav", "adindex", "ratingruneta", "alladvertising", "ruward",
        "workspace", "marketing-tech", "iguides", "ratingruneta",
        "rb.ru", "klerk", "cossa", "tadviser", "secret", "vc.",
        "dtf", "forbes", "rbc.ru", "kommersant", "vedomosti",
        "lenta", "gazeta", "ria", "tass", "kp.", "meduza", "rt.com",
        "sputnik", "interfax", "news", "media", "press", "journal",
        "tjournal",
    )):
        return "media"
    # Classifieds / marketplaces
    if any(k in h for k in (
        "avito", "cian", "drom", "auto.ru", "ozon", "wildberries",
        "ya.market", "market.yandex", "1seller", "insales", "rustore",
        "shop", "retail",
    )):
        return "classifieds"
    # Tech / UGC / dev
    if any(k in h for k in (
        "habr", "vc.ru", "pikabu", "github", "tproger", "rb.ru",
        "tenchat",
    )):
        return "tech"
    # Education / online courses
    if any(k in h for k in (
        "skillbox", "netology", "geekbrains", "yandex.practicum",
        "stepik", "lerna", ".edu", "university", "courses",
    )):
        return "education"
    # Video platforms
    if any(k in h for k in ("rutube", "youtube", "vimeo", "videopod")):
        return "video"
    # HR / jobs / services / reviews
    if any(k in h for k in (
        "dreamjob", "irecommend", "hh.ru", "superjob", "rabota",
        "hr.ru", "review", "otzov",
    )):
        return "services"
    # SaaS / CRM
    if any(k in h for k in ("crm", "saas", "platrum", "amocrm", "bitrix")):
        return "saas"
    # Maps / local
    if any(k in h for k in ("2gis", "maps.")):
        return "maps"
    # Academic / research
    if h.endswith(".ac.uk") or "cyberleninka" in h or "elibrary" in h or "scholar" in h:
        return "academic"
    # Reference
    if "wikipedia.org" in h:
        return "reference"
    # Dzen specifically (Yandex-owned but standalone domain)
    if h == "dzen.ru" or h.endswith(".dzen.ru"):
        return "yandex"
    return "other"


def main() -> None:
    if not CSV_PATH.exists():
        print(f"WARN: {CSV_PATH} not found - writing empty Set 2.")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({
            "id": "set2_corpus_sources",
            "title": "AI visibility research corpus sources (auto-extracted)",
            "description": "Source file not found. Place all_sources.csv into domens_collection/ and re-run scripts/extract_set2.py.",
            "extraction_meta": {
                "source_file": "domens_collection/all_sources.csv",
                "total_urls_found": 0,
                "total_urls_after_filter": 0,
                "unique_domains": 0,
                "filtered_domains_list": [],
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            },
            "domains": [],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    csv_size = CSV_PATH.stat().st_size
    print(f"Source: {CSV_PATH} ({csv_size:,} bytes)")

    raw_url_count = 0
    line_count = 0
    bad_url_count = 0
    counter: Counter[str] = Counter()

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        line_count += 1
        print(f"Header: {header}")
        for row in reader:
            line_count += 1
            if not row:
                continue
            url = row[0].strip()
            if not url:
                continue
            raw_url_count += 1
            try:
                parsed = urlparse(url)
            except Exception:
                bad_url_count += 1
                continue
            host = normalize_host(parsed.netloc)
            if not host:
                bad_url_count += 1
                continue
            counter[host] += 1

    print(f"Total CSV lines (incl. header): {line_count:,}")
    print(f"Total URL rows: {raw_url_count:,}")
    print(f"Bad/empty URLs skipped: {bad_url_count:,}")
    print(f"Unique hosts before filter: {len(counter):,}")

    junk_dropped: list[tuple[str, int]] = []
    for h in list(counter.keys()):
        if is_junk(h):
            junk_dropped.append((h, counter[h]))
            del counter[h]
    junk_dropped.sort(key=lambda x: -x[1])

    junk_url_total = sum(n for _, n in junk_dropped)
    print(f"Junk hosts dropped: {len(junk_dropped)} ({junk_url_total:,} URLs total)")
    for h, n in junk_dropped[:15]:
        print(f"  - {h}: {n:,}")
    if len(junk_dropped) > 15:
        print(f"  ... and {len(junk_dropped) - 15} more junk hosts")

    # TLD filter: keep only Russian-TLD domains.
    tld_dropped_hosts = 0
    tld_dropped_urls = 0
    for h in list(counter.keys()):
        if not has_allowed_tld(h):
            tld_dropped_urls += counter[h]
            tld_dropped_hosts += 1
            del counter[h]
    print(f"TLD filter: dropped {tld_dropped_hosts:,} non-RU hosts ({tld_dropped_urls:,} URLs)")
    print(f"  allowed TLDs: {list(ALLOWED_TLDS)}")

    # Frequency floor: long-tail entries with very few citations rarely move
    # the picture; drop them so the LLM analyzer stays focused on signal.
    freq_dropped_hosts = 0
    freq_dropped_urls = 0
    for h in list(counter.keys()):
        if counter[h] < MIN_FREQUENCY:
            freq_dropped_urls += counter[h]
            freq_dropped_hosts += 1
            del counter[h]
    print(f"Frequency filter: dropped {freq_dropped_hosts:,} hosts with <{MIN_FREQUENCY} citations ({freq_dropped_urls:,} URLs)")

    final_total_urls = sum(counter.values())
    print(f"Unique hosts after filter: {len(counter):,}")
    print(f"URLs accounted for after filter: {final_total_urls:,}")

    overrides = _load_overrides()
    print(f"\nLoaded {len(overrides):,} hand-curated category overrides from {OVERRIDES_PATH.name}")

    # build sorted domain list
    domains_out: list[dict] = []
    cat_counter: Counter[str] = Counter()
    overridden_count = 0
    for host, freq in counter.most_common():
        if host in overrides:
            cat = overrides[host]
            overridden_count += 1
        else:
            cat = category_guess(host)
        domains_out.append({"domain": host, "frequency": freq, "category_guess": cat})
        cat_counter[cat or "(uncategorised)"] += 1
    print(f"Categories applied via override: {overridden_count}/{len(domains_out)}")

    print("\nTop 30 by frequency:")
    for d in domains_out[:30]:
        cat = d["category_guess"] or "-"
        print(f"  {d['frequency']:>5}  {d['domain']:<40} {cat}")

    print("\nCategory distribution:")
    for cat, n in cat_counter.most_common():
        print(f"  {cat:<18} {n:>5}")
    null_count = sum(1 for d in domains_out if d['category_guess'] is None)
    print(f"\nDomains without a category: {null_count}/{len(domains_out)}")

    payload = {
        "id": "set2_corpus_sources",
        "title": "AI visibility research corpus sources (auto-extracted)",
        "description": "Домены, встречавшиеся как источники в предыдущих исследованиях AI visibility - то есть URL, которые web search агенты LLM-систем цитировали в ответах. Частота встречаемости отражает, насколько часто этот домен попадает в retrieval-ответы LLM. Filtered to Russian TLDs only (.ru, .рф, .su)",
        "extraction_meta": {
            "source_file": "domens_collection/all_sources.csv",
            "source_size_bytes": csv_size,
            "total_urls_found": raw_url_count,
            "total_urls_after_filter": final_total_urls,
            "unique_domains": len(domains_out),
            "filtered_domains_list": [h for h, _ in junk_dropped],
            "tld_filter_applied": True,
            "allowed_tlds": list(ALLOWED_TLDS),
            "dropped_by_tld_filter": tld_dropped_hosts,
            "min_frequency": MIN_FREQUENCY,
            "dropped_by_frequency_filter": freq_dropped_hosts,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        },
        "domains": domains_out,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
