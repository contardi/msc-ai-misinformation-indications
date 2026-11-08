"""Prepare the Data Commons ClaimReview feed into a local Brazilian fact-check corpus."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse as urlparse
from pathlib import Path

import httpx

FEED_URL = "https://storage.googleapis.com/datacommons-feeds/factcheck/latest/data.json"

_BR_HOSTS = ("aosfatos", "boatos", "e-farsas", "checamos.afp", "projetocomprova", "lupa.news")


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def _is_brazilian(publisher_url: str) -> bool:
    host = urlparse.urlparse(publisher_url or "").netloc.lower()
    return host.endswith(".br") or any(k in host for k in _BR_HOSTS)


def _reviews(item: dict) -> list[dict]:
    """A DataFeedItem's `item` may be a dict, a list, or null."""
    value = item.get("item")
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []


def load_feed(source: str | None) -> dict:
    if source:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        print(f"[claimreview] reading {path} ({path.stat().st_size / 1e6:.0f} MB)")
        return json.loads(path.read_bytes())

    print(f"[claimreview] downloading {FEED_URL}")
    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        resp = client.get(FEED_URL)
        resp.raise_for_status()
        print(f"[claimreview] {len(resp.content) / 1e6:.0f} MB downloaded")
        return json.loads(resp.content)


def normalise(feed: dict, brazilian_only: bool) -> list[dict]:
    elements = feed.get("dataFeedElement") or []
    print(f"[claimreview] feed dated {feed.get('dateModified', '?')}, {len(elements)} elements")

    rows, no_review, no_claim = [], 0, 0
    for element in elements:
        reviews = _reviews(element)
        if not reviews:
            no_review += 1
            continue
        for review in reviews:
            author = review.get("author") or {}
            publisher_url = author.get("url") or ""
            if brazilian_only and not _is_brazilian(publisher_url):
                continue
            claim = (review.get("claimReviewed") or "").strip()
            if not claim:
                no_claim += 1
                continue
            rows.append({
                "claim": " ".join(claim.split()),
                "rating": ((review.get("reviewRating") or {}).get("alternateName") or "").strip(),
                "publisher": (author.get("name") or "").strip(),
                "publisher_url": publisher_url,
                "url": review.get("url") or "",
                "date": (review.get("datePublished") or "")[:10],
            })
    print(f"[claimreview] skipped: {no_review} elements with no review, {no_claim} with no claim text")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Filter the ClaimReview feed to a local corpus.")
    p.add_argument("--json", default=None, help="Local copy of the feed instead of downloading.")
    p.add_argument("--out", default=None, help="Output CSV (default: <DATA_DIR>/processed/claimreview_br.csv).")
    p.add_argument("--all-languages", action="store_true", help="Do not filter to Brazilian publishers.")
    args = p.parse_args(argv)

    try:
        feed = load_feed(args.json)
    except Exception as exc:  # noqa: BLE001 - fail loudly; never write a truncated corpus
        print(f"[claimreview] ERROR: could not load the feed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    rows = normalise(feed, brazilian_only=not args.all_languages)
    if not rows:
        print("[claimreview] ERROR: no rows matched. Refusing to write an empty corpus.",
              file=sys.stderr)
        return 1

    by_publisher: dict[str, int] = {}
    for r in rows:
        by_publisher[r["publisher"] or "(unnamed)"] = by_publisher.get(r["publisher"] or "(unnamed)", 0) + 1
    print(f"\n[claimreview] {len(rows)} checks from {len(by_publisher)} publishers")
    for name, n in sorted(by_publisher.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {n:>6}  {name}")

    with_url = sum(1 for r in rows if r["url"])
    dated = sum(1 for r in rows if r["date"])
    print(f"\n[claimreview] {with_url}/{len(rows)} have a URL, {dated}/{len(rows)} have a date")

    out_path = Path(args.out) if args.out else _data_dir() / "processed" / "claimreview_br.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[claimreview] wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
