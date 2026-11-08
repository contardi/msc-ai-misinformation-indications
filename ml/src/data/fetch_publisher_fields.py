"""Fetch publishers' own title AND excerpt for corpus rows we already hold, to answer one question."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import urllib.parse as up
from pathlib import Path

import httpx

csv.field_size_limit(10**9)

USER_AGENT = (
    "misinformation-research-probe/1.0 (MSc dissertation, University of Essex Online; "
    "title lookup for a retrieval experiment; contact via repository)"
)

SOURCES = {
    "boatos.org": ("https://www.boatos.org/wp-json/wp/v2/posts", "Boatos.org"),
    "lupa.uol.com.br": ("https://lupa.uol.com.br/wp-json/wp/v2/posts", "Agência Lupa"),
}

BATCH = 20
DELAY = 0.35


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/app/data")) / "processed"


def slug_of(url: str) -> str:
    return up.urlparse(url).path.strip("/").split("/")[-1].removesuffix(".html")


_VERDICT_PREFIX = re.compile(r"^\s*(boato|bulo|rumor|hoax|fake|fato)\s*[\u2013\u2014-]\s*", re.IGNORECASE)


def split_verdict(excerpt: str) -> tuple[str, str]:
    """(verdict_word, remaining_text)."""
    m = _VERDICT_PREFIX.match(excerpt)
    if not m:
        return "", excerpt.strip()
    return m.group(1), excerpt[m.end():].strip()


def fetch_fields(client: httpx.Client, base: str, slugs: list[str]) -> dict[str, dict]:
    """slug -> {title, excerpt} for as many as the API returns."""
    out: dict[str, dict] = {}
    for i in range(0, len(slugs), BATCH):
        chunk = slugs[i:i + BATCH]
        query = "&".join(f"slug[]={up.quote(s)}" for s in chunk)
        try:
            r = client.get(f"{base}?{query}&_fields=slug,title,excerpt&per_page={BATCH}")
        except Exception as exc:  # noqa: BLE001
            print(f"    request failed ({type(exc).__name__}); continuing", file=sys.stderr)
            time.sleep(DELAY * 4)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            print(f"    HTTP {r.status_code}; backing off", file=sys.stderr)
            time.sleep(DELAY * 8)
            continue
        if r.status_code != 200:
            continue
        try:
            items = r.json()
        except ValueError:
            continue
        for item in items:
            title = strip_html((item.get("title") or {}).get("rendered", ""))
            excerpt = strip_html((item.get("excerpt") or {}).get("rendered", ""))
            if item.get("slug") and title:
                out[item["slug"]] = {"title": title, "excerpt": excerpt}
        done = min(i + BATCH, len(slugs))
        print(f"    {done}/{len(slugs)} slugs, {len(out)} resolved", end="\r", file=sys.stderr)
        time.sleep(DELAY)
    print(file=sys.stderr)
    return out


def strip_html(text: str) -> str:
    """WordPress returns rendered HTML with entities in both `title` and `excerpt`."""
    import html

    return html.unescape(re.sub(r"<[^>]+>", " ", text or "")).replace("\xa0", " ").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch publisher titles for existing corpus rows.")
    parser.add_argument("--limit", type=int, default=0, help="Cap rows per publisher (0 = all).")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    src = _data_dir() / "claimreview_br.csv"
    if not src.exists():
        print(f"[titles] {src} not found")
        return 1

    rows: list[dict] = []
    with open(src, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())

    by_source: dict[str, list[dict]] = {k: [] for k in SOURCES}
    for row in rows:
        host = up.urlparse(row["url"]).netloc.lower()
        for key in SOURCES:
            if key in host:
                by_source[key].append(row)
                break

    fields: dict[str, dict] = {}
    verdicts: dict[str, int] = {}
    with httpx.Client(timeout=args.timeout, follow_redirects=True,
                      headers={"User-Agent": USER_AGENT}) as client:
        for key, (base, label) in SOURCES.items():
            subset = by_source[key]
            if args.limit:
                subset = subset[: args.limit]
            if not subset:
                continue
            print(f"[fields] {label}: {len(subset)} rows")
            slugs = [slug_of(r["url"]) for r in subset]
            found = fetch_fields(client, base, slugs)
            for row, slug in zip(subset, slugs):
                if slug in found:
                    fields[row["url"]] = found[slug]
            with_verdict = sum(1 for s_ in found.values() if split_verdict(s_["excerpt"])[0])
            verdicts[label] = with_verdict
            print(f"[fields] {label}: {len(found)}/{len(subset)} resolved, "
                  f"{with_verdict} carry a verdict prefix in the excerpt")

    matched = [r for r in rows if r["url"] in fields]
    if not matched:
        print("[fields] nothing resolved; not writing")
        return 1

    variants = {
        "matched": lambda row: row["claim"],
        "titles": lambda row: fields[row["url"]]["title"],
        "excerpts": lambda row: split_verdict(fields[row["url"]]["excerpt"])[1]
                                or fields[row["url"]]["title"],
    }
    for name, pick in variants.items():
        path = _data_dir() / f"claimreview_br_{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in matched:
                out = dict(row)
                out["claim"] = pick(row)
                writer.writerow(out)
        print(f"[fields] wrote {path} ({len(matched)} rows)")

    total = sum(len(v) for v in by_source.values())
    print(f"\n[fields] {len(matched)}/{total} rows resolved.")
    for label, n in verdicts.items():
        print(f"[fields]   {label}: {n} excerpts carry a quotable verdict word")
    print("[fields] Measure all three with the production harness, one variable changed each time:")
    for name in variants:
        print(f"  CLAIMREVIEW_INDEX=/app/data/processed/claimreview_br_{name}.csv  ... --sources local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
