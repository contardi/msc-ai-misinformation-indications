"""Harvest published fact-checks from publishers' own APIs into the Postgres cache."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone

import httpx

sys.path.insert(0, "/app")

from app.api.verdicts import canonical_verdict  # noqa: E402

USER_AGENT = (
    "misinformation-research-harvester/1.0 (MSc dissertation, University of Essex Online; "
    "indexes published fact-check metadata for a search interface; contact via repository)"
)

PER_PAGE = 100
DELAY = 0.4
BACKOFF = 5.0

PUBLISHERS: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    "boatos": ("https://www.boatos.org/wp-json/wp/v2/posts", "title", ("english", "espanol"), "wordpress"),
    "lupa": ("https://lupa.uol.com.br/wp-json/wp/v2/posts", "title", (), "wordpress"),
    "comprova": ("https://projetocomprova.com.br/wp-json/wp/v2/publicacoes", "title", (), "wordpress"),
    "g1": ("https://g1.globo.com/rss/g1/fato-ou-fake/", "title", (), "rss"),
}

G1_MARKERS: tuple[tuple[str, str], ...] = (
    ("#NÃOÉBEMASSIM", r"^\s*#\s*N[ÃA]O\s*[ÉE]\s*BEM\s*ASSIM\b"),
    ("#FAKE", r"^\s*(?:É\s+)?#?\s*FAKE\b"),
    ("#FATO", r"^\s*(?:É\s+)?#?\s*FATO\b(?!\s*(?:ou|/)\s*#?\s*FAKE)"),
)
_G1_MARKERS = tuple((label, re.compile(pat, re.IGNORECASE)) for label, pat in G1_MARKERS)
_G1_LEAD_QUE = re.compile(r"^\s*que\s+", re.IGNORECASE)


def g1_marker(title: str) -> tuple[str, str] | None:
    """(the publisher's verdict marker, the claim with the marker removed), or None to refuse."""
    for label, pattern in _G1_MARKERS:
        m = pattern.match(title or "")
        if m:
            claim = _G1_LEAD_QUE.sub("", (title or "")[m.end():]).strip(" -–—:;,")
            return label, claim
    return None


def excluded_by_path(url: str, exclude: tuple[str, ...]) -> bool:
    """True when the URL's first path segment marks a non-Brazilian edition."""
    if not exclude:
        return False
    from urllib.parse import urlparse

    first = urlparse(url).path.strip("/").split("/")[0].lower()
    return first in exclude

_VERDICT_PREFIX = re.compile(r"^\s*(boato|bulo|rumor|hoax|fake|fato)\s*[–—-]\s*", re.I)
_TAG = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    import html

    return html.unescape(_TAG.sub(" ", text or "")).replace("\xa0", " ").strip()


def split_verdict(excerpt: str) -> tuple[str, str]:
    """(the publisher's verdict word, the remaining claim text)."""
    m = _VERDICT_PREFIX.match(excerpt)
    return (m.group(1), excerpt[m.end():].strip()) if m else ("", excerpt.strip())


def connect():
    import psycopg

    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit("[harvest] DATABASE_URL is not set")
    return psycopg.connect(url)


def watermark(conn, slug: str) -> datetime | None:
    """The newest item already harvested, so a refresh asks only for what is new."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT max(newest_item_at) FROM harvest_runs WHERE publisher_slug = %s AND error = ''",
            (slug,),
        )
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def fetch_page(client: httpx.Client, endpoint: str, page: int, after: datetime | None) -> tuple[list, int]:
    params = {"per_page": PER_PAGE, "page": page, "_fields": "date_gmt,link,title,excerpt"}
    if after:
        params["after"] = after.astimezone(timezone.utc).replace(tzinfo=None).isoformat()
    r = client.get(endpoint, params=params)
    if r.status_code in (429, 500, 502, 503, 504):
        time.sleep(BACKOFF)
        return [], r.status_code
    if r.status_code == 400:
        return [], 200
    if r.status_code != 200:
        return [], r.status_code
    try:
        return r.json(), 200
    except ValueError:
        return [], r.status_code


def fetch_rss(client: httpx.Client, endpoint: str) -> tuple[list[dict], int]:
    """The feed's items as plain dicts."""
    import xml.etree.ElementTree as ET

    r = client.get(endpoint)
    if r.status_code in (429, 500, 502, 503, 504):
        time.sleep(BACKOFF)
        return [], r.status_code
    if r.status_code != 200:
        return [], r.status_code
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return [], r.status_code

    items = []
    for node in root.findall(".//item"):
        items.append({
            "title": (node.findtext("title") or "").strip(),
            "link": (node.findtext("link") or "").strip(),
            "pubDate": (node.findtext("pubDate") or "").strip(),
            "description": node.findtext("description") or "",
        })
    return items, 200


def parse_rss_date(raw: str) -> datetime | None:
    """RFC 822, which is what RSS uses and datetime.fromisoformat cannot read."""
    from email.utils import parsedate_to_datetime

    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def rss_to_row(item: dict, publisher_slug: str, publisher_name: str) -> dict | None:
    """One feed item as a row, or None to refuse it."""
    url = (item.get("link") or "").strip()
    title = strip_html(item.get("title") or "")
    if not url or not title:
        return None

    marked = g1_marker(title)
    if marked is None:
        return None
    rating, claim = marked
    if not claim:
        return None

    return {
        "url": url,
        "claim": claim,
        "claim_is_title": True,
        "summary": strip_html(item.get("description") or "") or None,
        "rating": rating,
        "verdict": canonical_verdict(rating),
        "publisher_slug": publisher_slug,
        "publisher": publisher_name,
        "published_at": parse_rss_date(item.get("pubDate") or ""),
        "source": "rss",
    }


def to_row(item: dict, publisher_slug: str, claim_field: str, publisher_name: str,
           exclude: tuple[str, ...] = ()) -> dict | None:
    url = (item.get("link") or "").strip()
    if excluded_by_path(url, exclude):
        return None
    title = strip_html((item.get("title") or {}).get("rendered", ""))
    excerpt = strip_html((item.get("excerpt") or {}).get("rendered", ""))
    if not url or not title:
        return None

    rating, body = split_verdict(excerpt)
    claim = body if claim_field == "excerpt" and body else title
    if not claim:
        return None

    published = None
    raw = (item.get("date_gmt") or "").strip()
    if raw:
        try:
            published = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
        except ValueError:
            published = None

    return {
        "url": url,
        "claim": claim,
        "claim_is_title": claim_field != "excerpt" or not body,
        "rating": rating,
        "verdict": canonical_verdict(rating),
        "publisher_slug": publisher_slug,
        "publisher": publisher_name,
        "published_at": published,
        "source": "wordpress",
        "summary": None,
    }


def upsert(conn, rows: list[dict]) -> int:
    """Idempotent on `url`."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO fact_checks (url, claim, claim_is_title, summary, rating, verdict,
                                     publisher_slug, publisher, published_at, source, fetched_at)
            VALUES (%(url)s, %(claim)s, %(claim_is_title)s, %(summary)s, %(rating)s, %(verdict)s,
                    %(publisher_slug)s, %(publisher)s, %(published_at)s, %(source)s, NOW())
            ON CONFLICT (url) DO UPDATE SET
                claim = EXCLUDED.claim, claim_is_title = EXCLUDED.claim_is_title,
                summary = EXCLUDED.summary,
                rating = EXCLUDED.rating, verdict = EXCLUDED.verdict,
                publisher_slug = EXCLUDED.publisher_slug, publisher = EXCLUDED.publisher,
                published_at = EXCLUDED.published_at, source = EXCLUDED.source, fetched_at = NOW()
            """,
            rows,
        )
    return len(rows)


def harvest(conn, client: httpx.Client, slug: str, max_pages: int, dry_run: bool,
            full: bool = False) -> dict:
    endpoint, claim_field, exclude, mechanism = PUBLISHERS[slug]
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM publishers WHERE slug = %s", (slug,))
        row = cur.fetchone()
    name = row[0] if row else slug

    after = None if full else watermark(conn, slug)
    mode = ("full backfill (--full, ignoring the watermark)" if full
            else f"incremental since {after.date().isoformat()}" if after
            else "full backfill (no watermark yet)")
    print(f"[harvest] {name}: via={mechanism} field={claim_field} {mode}")

    seen = added = 0
    newest: datetime | None = None
    status = 200
    error = ""
    pages = range(1, 2) if mechanism == "rss" else range(1, max_pages + 1)
    for page in pages:
        if mechanism == "rss":
            items, status = fetch_rss(client, endpoint)
        else:
            items, status = fetch_page(client, endpoint, page, after)
        if status != 200:
            error = f"HTTP {status} on page {page}"
            break
        if not items:
            break
        if mechanism == "rss":
            built = [rss_to_row(i, slug, name) for i in items]
            rows = [r for r in built if r
                    and (after is None or r["published_at"] is None or r["published_at"] > after)]
        else:
            rows = [r for r in (to_row(i, slug, claim_field, name, exclude) for i in items) if r]
        seen += len(items)
        for r in rows:
            if r["published_at"] and (newest is None or r["published_at"] > newest):
                newest = r["published_at"]
        if not dry_run:
            added += upsert(conn, rows)
        print(f"    page {page}: {len(items)} items, {len(rows)} kept "
              f"({len(items) - len(rows)} filtered), {seen} seen", end="\r")
        time.sleep(DELAY)
    print()

    if not dry_run:
        cur_newest = None if error else newest
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO harvest_runs (publisher_slug, finished_at, newest_item_at,
                                             items_seen, items_added, http_status, error)
                   VALUES (%s, NOW(), %s, %s, %s, %s, %s)""",
                (slug, cur_newest, seen, added, status, error),
            )
        conn.commit()
    return {"publisher": name, "seen": seen, "added": added, "newest": newest, "error": error}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harvest published fact-checks into Postgres.")
    parser.add_argument("--publishers", nargs="*", default=list(PUBLISHERS),
                        help=f"Any of: {', '.join(PUBLISHERS)}")
    parser.add_argument("--max-pages", type=int, default=200,
                        help="Safety bound. 200 pages x 100 = 20,000 items per publisher.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and report, write nothing. Run this first.")
    parser.add_argument("--full", action="store_true",
                        help="Ignore the watermark and re-walk everything. Needed after a partial "
                             "run, which otherwise leaves a watermark that blocks the backfill.")
    args = parser.parse_args(argv)

    unknown = [p for p in args.publishers if p not in PUBLISHERS]
    if unknown:
        print(f"[harvest] unknown publisher(s): {unknown}; known: {list(PUBLISHERS)}")
        return 1

    results = []
    with connect() as conn, httpx.Client(timeout=30, follow_redirects=True,
                                         headers={"User-Agent": USER_AGENT}) as client:
        for slug in args.publishers:
            try:
                results.append(harvest(conn, client, slug, args.max_pages, args.dry_run, args.full))
            except Exception as exc:  # noqa: BLE001 - one publisher must not stop the rest
                print(f"[harvest] {slug} failed: {type(exc).__name__}: {exc}")
                results.append({"publisher": slug, "seen": 0, "added": 0,
                                "newest": None, "error": str(exc)})

        if not args.dry_run:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM fact_checks")
                total = cur.fetchone()[0]
        else:
            total = None

    print(f"\n{'publisher':20s} {'seen':>7s} {'added':>7s}  newest")
    for r in results:
        newest = r["newest"].date().isoformat() if r["newest"] else "-"
        print(f"{r['publisher']:20s} {r['seen']:7d} {r['added']:7d}  {newest}"
              + (f"   ERROR: {r['error']}" if r["error"] else ""))
    if total is not None:
        print(f"\n[harvest] fact_checks now holds {total} rows")
    if args.dry_run:
        print("[harvest] dry run: nothing was written. Drop --dry-run to harvest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
