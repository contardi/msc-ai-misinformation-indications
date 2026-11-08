"""Do the style cues shown to users actually indicate misinformation?"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/app")

from app.explanation.cues import _SENSATIONAL, extract_cues  # noqa: E402

csv.field_size_limit(10**9)

ARTICLE_CORPORA = ("fakebr", "fakerecogna2")

ANY_CUE = "any_cue"


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/app/data")) / "processed"


def _results() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "/app/results"))


def wilson_lower(hits: int, total: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson interval for a proportion."""
    if total == 0:
        return 0.0
    p = hits / total
    d = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, (centre - margin) / d)


def rows_from_corpus(path: Path, category: str | None = None):
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if category and (row.get("category") or "").strip().lower() != category:
                continue
            text, label = row.get("text"), row.get("label")
            if text and label in ("0", "1"):
                yield text, int(label)


def short_text_rows(seed: int = 42):
    """188 false claims from 2025-26 against matched real headlines."""
    claims_path = _results() / "current_claims.csv"
    corpus_path = _data_dir() / "fakerecogna2.csv"
    if not claims_path.exists() or not corpus_path.exists():
        return []
    positives = []
    with open(claims_path, newline="", encoding="utf-8") as fh:
        positives = [r["claim"] for r in csv.DictReader(fh) if r.get("claim")]
    titles = []
    with open(corpus_path, newline="", encoding="utf-8") as fh:
        titles = [r["title"].strip() for r in csv.DictReader(fh)
                  if r.get("label") == "0" and (r.get("title") or "").strip()]
    if not positives or len(titles) < len(positives):
        return []
    negatives = random.Random(seed).sample(titles, len(positives))
    return [(t, 1) for t in positives] + [(t, 0) for t in negatives]


def measure(rows) -> tuple[dict, dict, dict[int, int]]:
    """Returns (per-cue counts, per-term counts, class totals)."""
    cues: dict[str, dict[int, int]] = {}
    terms: dict[str, dict[int, int]] = {}
    totals = {0: 0, 1: 0}
    for text, label in rows:
        totals[label] += 1
        found = extract_cues(text)
        for ind in found:
            cues.setdefault(ind.kind, {0: 0, 1: 0})[label] += 1
            for term in ind.terms:
                terms.setdefault(term, {0: 0, 1: 0})[label] += 1
        if found:
            cues.setdefault(ANY_CUE, {0: 0, 1: 0})[label] += 1
    return cues, terms, totals


def _report(name: str, cues: dict, terms: dict, totals: dict[int, int], out: list[dict]) -> None:
    n_fake, n_real = totals[1], totals[0]
    base = n_fake / max(n_fake + n_real, 1)
    print(f"\n=== {name} ===  real={n_real}  fake={n_fake}")
    if not n_fake or not n_real:
        return
    print(f"  {'cue':20s} {'fake%':>7s} {'real%':>7s} {'lift':>6s} {'precis':>7s} {'wilson':>7s} {'n':>7s}")
    for kind, counts in sorted(cues.items(), key=lambda kv: -kv[1][1] / n_fake):
        fired = counts[0] + counts[1]
        fr, rr = counts[1] / n_fake * 100, counts[0] / n_real * 100
        prec = counts[1] / fired * 100 if fired else 0.0
        lift = (prec / 100) / base if base else 0.0
        wl = wilson_lower(counts[1], fired) * 100
        flag = "  <-- fires MORE on legitimate text" if lift < 1.0 else ""
        print(f"  {kind:20s} {fr:7.1f} {rr:7.1f} {lift:6.2f} {prec:7.1f} {wl:7.1f} {fired:7d}{flag}")
        out.append({"corpus": name, "kind": kind, "term": "", "n_fired": fired,
                    "fake_rate": round(fr, 2), "real_rate": round(rr, 2), "lift": round(lift, 3),
                    "precision": round(prec, 2), "wilson_lower": round(wl, 2)})
    if not terms:
        return
    print(f"  -- per sensational term (of {len(_SENSATIONAL)} in the list) --")
    for term, counts in sorted(terms.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        fired = counts[0] + counts[1]
        prec = counts[1] / fired * 100
        lift = (prec / 100) / base if base else 0.0
        wl = wilson_lower(counts[1], fired) * 100
        flag = "  <-- no better than chance" if wl < base * 100 else ""
        print(f"    {term:16s} n={fired:6d}  fake={counts[1]:6d} real={counts[0]:6d}  "
              f"precision={prec:5.1f}%  wilson>={wl:5.1f}%  lift={lift:.2f}{flag}")
        out.append({"corpus": name, "kind": "sensational_terms", "term": term, "n_fired": fired,
                    "fake_rate": round(counts[1] / n_fake * 100, 2),
                    "real_rate": round(counts[0] / n_real * 100, 2), "lift": round(lift, 3),
                    "precision": round(prec, 2), "wilson_lower": round(wl, 2)})
    never = sorted(t for t in _SENSATIONAL if t not in terms)
    if never:
        print(f"    never fired: {', '.join(never)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the user-facing style cues against labels.")
    parser.add_argument("--corpora", nargs="*", default=list(ARTICLE_CORPORA))
    parser.add_argument("--out", default="cue_precision.csv", help="Written under RESULTS_DIR.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Write even when a corpus is missing. Pair it with --out to a scratch "
                             "filename; the default target is committed evidence.")
    args = parser.parse_args(argv)

    out: list[dict] = []
    skipped: list[str] = []
    for corpus in args.corpora:
        path = _data_dir() / f"{corpus}.csv"
        if not path.exists():
            print(f"\n=== {corpus} === (skipped: {path} not found)")
            skipped.append(corpus)
            continue
        _report(corpus, *measure(rows_from_corpus(path)), out)
        if corpus == "fakerecogna2":
            _report(f"{corpus} [politica]", *measure(rows_from_corpus(path, "política")), out)

    short = short_text_rows()
    if short:
        _report("short text (2025-26 claims vs headlines)", *measure(short), out)
    else:
        print("\n=== short text === (skipped: current_claims.csv or fakerecogna2.csv not found)")

    if skipped and not args.allow_partial:
        print(f"\nREFUSING TO WRITE: {', '.join(skipped)} unavailable, so these results are partial.")
        print("Writing them would overwrite committed evidence with a degraded measurement.")
        print("Fetch the missing corpora, or pass --allow-partial with --out to a scratch filename.")
        return 1

    if out:
        dest = _results() / args.out
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(out)
        print(f"\nsaved {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
