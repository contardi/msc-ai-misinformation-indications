"""Does the harvest surface checks for claims circulating NOW?"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from app.api import db_factcheck, local_factcheck  # noqa: E402

csv.field_size_limit(10**9)


def _results() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "/app/results"))


def _data() -> Path:
    return Path(os.environ.get("DATA_DIR", "/app/data")) / "processed"


def load_sets(seed: int) -> tuple[list[str], list[str]]:
    claims_path = _results() / "current_claims.csv"
    corpus_path = _data() / "fakerecogna2.csv"
    if not claims_path.exists() or not corpus_path.exists():
        return [], []
    with open(claims_path, newline="", encoding="utf-8") as fh:
        positives = [r["claim"] for r in csv.DictReader(fh) if r.get("claim")]
    with open(corpus_path, newline="", encoding="utf-8") as fh:
        titles = [r["title"].strip() for r in csv.DictReader(fh)
                  if r.get("label") == "0" and (r.get("title") or "").strip()]
    if not positives or len(titles) < len(positives):
        return positives, []
    return positives, random.Random(seed).sample(titles, len(positives))


def rate(texts: list[str], search) -> tuple[int, float]:
    hits = sum(1 for t in texts if search(t, 5))
    return hits, (hits / len(texts) * 100 if texts else 0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval for claims circulating in 2025-26.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="current-retrieval.csv")
    args = parser.parse_args(argv)

    positives, controls = load_sets(args.seed)
    if not positives or not controls:
        print("[current] missing current_claims.csv or fakerecogna2.csv; nothing measured")
        return 1

    sources = [
        ("bundled CSV (stops in 2024)", local_factcheck.search_local, local_factcheck.is_enabled()),
        ("harvested database", db_factcheck.search_db, db_factcheck.is_enabled()),
    ]
    print(f"[current] {len(positives)} claims from 2025-26 vs {len(controls)} real headlines\n")
    print(f"  {'source':32s} {'found':>8s} {'false assoc':>13s}")

    rows = []
    for name, search, enabled in sources:
        if not enabled:
            print(f"  {name:32s} {'disabled':>8s}")
            continue
        p_hits, p_rate = rate(positives, search)
        c_hits, c_rate = rate(controls, search)
        print(f"  {name:32s} {p_rate:7.1f}% {c_rate:12.1f}%")
        rows.append({"source": name, "positives_n": len(positives), "positives_found": p_hits,
                     "positives_rate": round(p_rate / 100, 4), "controls_n": len(controls),
                     "controls_matched": c_hits, "controls_rate": round(c_rate / 100, 4)})

    def both(text, limit=5):
        seen, out = set(), []
        for _, search, enabled in sources:
            if not enabled:
                continue
            for fc in search(text, limit):
                if fc.url not in seen:
                    seen.add(fc.url)
                    out.append(fc)
        return out[:limit]

    p_hits, p_rate = rate(positives, both)
    c_hits, c_rate = rate(controls, both)
    print(f"  {'BOTH (what a user gets)':32s} {p_rate:7.1f}% {c_rate:12.1f}%")
    rows.append({"source": "both", "positives_n": len(positives), "positives_found": p_hits,
                 "positives_rate": round(p_rate / 100, 4), "controls_n": len(controls),
                 "controls_matched": c_hits, "controls_rate": round(c_rate / 100, 4)})

    if rows:
        dest = _results() / args.out
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"\n[current] saved {dest}")
    print("[current] Controls are 2020-22 headlines, so era is confounded with the label - a false")
    print("[current] association here is meaningful, a low positive rate is an upper bound only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
