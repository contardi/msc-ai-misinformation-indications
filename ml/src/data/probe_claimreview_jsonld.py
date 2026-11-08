"""Which Brazilian fact-checkers publish ClaimReview markup on their own article pages?"""

from __future__ import annotations

import argparse
import json
import re
import sys

import httpx

USER_AGENT = (
    "misinformation-research-probe/1.0 (MSc dissertation, University of Essex Online; "
    "one-off ClaimReview markup probe; contact via repository)"
)

PUBLISHERS: list[tuple[str, str, str]] = [
    ("Boatos.org", "https://www.boatos.org/wp-json/wp/v2/posts?per_page=1&_fields=link", "wp-json"),
    ("Agência Lupa", "https://lupa.uol.com.br/wp-json/wp/v2/posts?per_page=1&_fields=link", "wp-json"),
    ("Projeto Comprova",
     "https://projetocomprova.com.br/wp-json/wp/v2/publicacoes?per_page=1&_fields=link", "wp-json"),
    ("e-Farsas", "https://www.e-farsas.com/feed", "rss"),
    ("Aos Fatos", "https://www.aosfatos.org/noticias/feed/", "rss"),
    ("Estadão Verifica", "https://www.estadao.com.br/arc/outboundfeeds/sitemap-news/", "rss"),
    ("AFP Checamos", "https://checamos.afp.com/rss.xml", "rss"),
]

_LDJSON = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
_RSS_LINK = re.compile(r"<link>\s*(https?://[^<\s]+)\s*</link>", re.IGNORECASE)


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9"},
    )


def find_article(client: httpx.Client, listing: str, kind: str) -> tuple[str | None, str]:
    """One recent article URL from the publisher's own listing, or (None, reason)."""
    try:
        r = client.get(listing)
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, it does not raise
        return None, f"{type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return None, f"listing HTTP {r.status_code}"
    if kind == "wp-json":
        try:
            items = r.json()
        except ValueError:
            return None, "listing was not JSON"
        if not items:
            return None, "listing empty"
        return items[0].get("link"), ""
    links = [u for u in _RSS_LINK.findall(r.text) if "/feed" not in u]
    return (links[1] if len(links) > 1 else (links[0] if links else None)), (
        "" if links else "no <link> in feed"
    )


def claimreview_blocks(html: str) -> list[dict]:
    """Every ClaimReview object in the page's JSON-LD."""
    found: list[dict] = []

    def walk(node) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            types = node.get("@type", "")
            types = types if isinstance(types, list) else [types]
            if any(str(t).endswith("ClaimReview") for t in types):
                found.append(node)
            for key in ("@graph", "itemListElement", "mainEntity"):
                if key in node:
                    walk(node[key])

    for raw in _LDJSON.findall(html):
        try:
            walk(json.loads(raw.strip()))
        except ValueError:
            continue
    return found


def probe(client: httpx.Client, name: str, listing: str, kind: str) -> dict:
    url, why = find_article(client, listing, kind)
    if not url:
        return {"publisher": name, "article": None, "claimreview": False, "note": why}
    try:
        page = client.get(url)
    except Exception as exc:  # noqa: BLE001
        return {"publisher": name, "article": url, "claimreview": False,
                "note": f"{type(exc).__name__}: {exc}"}
    if page.status_code != 200:
        return {"publisher": name, "article": url, "claimreview": False,
                "note": f"article HTTP {page.status_code}"}

    blocks = claimreview_blocks(page.text)
    if not blocks:
        return {"publisher": name, "article": url, "claimreview": False,
                "note": "no ClaimReview JSON-LD on the page"}

    block = blocks[0]
    rating = block.get("reviewRating") or {}
    return {
        "publisher": name,
        "article": url,
        "claimreview": True,
        "claimReviewed": (block.get("claimReviewed") or "")[:110],
        "rating": rating.get("alternateName") or rating.get("ratingValue") or "",
        "datePublished": block.get("datePublished", ""),
        "note": f"{len(blocks)} ClaimReview block(s)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe publishers for on-page ClaimReview JSON-LD.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args(argv)

    results = []
    with _client(args.timeout) as client:
        for name, listing, kind in PUBLISHERS:
            results.append(probe(client, name, listing, kind))

    if args.json:
        json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    print(f"{'publisher':20s} {'JSON-LD':>8s}  detail")
    for r in results:
        mark = "YES" if r["claimreview"] else "no"
        print(f"{r['publisher']:20s} {mark:>8s}  {r['note']}")
        if r["claimreview"]:
            print(f"{'':20s} {'':>8s}  claimReviewed: {r.get('claimReviewed', '')!r}")
            print(f"{'':20s} {'':>8s}  rating: {r.get('rating', '')!r}  "
                  f"date: {r.get('datePublished', '')}")

    yes = [r["publisher"] for r in results if r["claimreview"]]
    no = [r["publisher"] for r in results if not r["claimreview"]]
    print(f"\n  real claimReviewed available : {', '.join(yes) or 'none'}")
    print(f"  titles only, must be measured: {', '.join(no) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
