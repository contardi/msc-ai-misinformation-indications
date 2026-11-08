"""Is retrieval weak because coverage is thin, or because our query matching is poor?"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_factcheck_retrieval import _tokens, load_sample  # noqa: E402

SWEEP = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
MIN_SHARED = 4


def _results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "/app/results"))


def load_corpus(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("claim") or "").strip()]
    for r in rows:
        r["_tok"] = _tokens(r["claim"])
    return [r for r in rows if r["_tok"]]


def build_index(corpus: list[dict]) -> dict[str, list[int]]:
    """Inverted index token -> corpus rows."""
    index: dict[str, list[int]] = collections.defaultdict(list)
    for i, row in enumerate(corpus):
        for tok in row["_tok"]:
            index[tok].append(i)
    return index


def best_match(ref: set[str], corpus: list[dict], index: dict[str, list[int]]) -> tuple[float, dict | None]:
    """Best overlap against the corpus, requiring MIN_SHARED absolute tokens as well as a ratio."""
    if not ref:
        return 0.0, None
    candidates: set[int] = set()
    for tok in ref:
        candidates.update(index.get(tok, ()))
    best_score, best_row = 0.0, None
    for i in candidates:
        other = corpus[i]["_tok"]
        shared = len(ref & other)
        if shared < MIN_SHARED:
            continue
        score = shared / min(len(ref), len(other))
        if score > best_score:
            best_score, best_row = score, corpus[i]
    return best_score, best_row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Coverage ceiling from the local ClaimReview corpus.")
    p.add_argument("--corpus", default="/app/data/processed/claimreview_br.csv")
    p.add_argument("--data", default="/app/data/processed/fakerecogna2.csv")
    p.add_argument("--sample", type=int, default=200,
                   help="Must match the retrieval run's sample so the two are comparable.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"[coverage] {corpus_path} not found. Build it first:\n"
              "  python ml/src/data/prepare_claimreview_feed.py")
        return 1

    corpus = load_corpus(corpus_path)
    index = build_index(corpus)
    print(f"[coverage] corpus: {len(corpus)} Brazilian checks, {len(index)} distinct terms")

    fakes, reals = load_sample(Path(args.data), args.sample, args.seed)
    print(f"[coverage] articles: {len(fakes)} known-fake (positives), {len(reals)} real (controls)")

    out_rows = []
    for group, records in (("fake", fakes), ("real", reals)):
        for rec in records:
            ref = _tokens(rec.get("title") or rec["text"][:300])
            score, row = best_match(ref, corpus, index)
            out_rows.append({
                "group": group,
                "score": round(score, 3),
                "publisher": (row or {}).get("publisher", ""),
                "matched_claim": (row or {}).get("claim", "")[:200],
            })

    fake_scores = [r["score"] for r in out_rows if r["group"] == "fake"]
    real_scores = [r["score"] for r in out_rows if r["group"] == "real"]

    print("\n" + "=" * 84)
    print("Threshold sweep - the REAL column is the control that makes any of this readable")
    print("=" * 84)
    print(f"  {'thr':>5} {'fake matched':>14} {'real matched':>14} {'gap':>8}")
    for t in SWEEP:
        f = sum(1 for s_ in fake_scores if s_ >= t) / len(fake_scores)
        r = sum(1 for s_ in real_scores if s_ >= t) / len(real_scores)
        print(f"  {t:>5.2f} {f:>13.1%} {r:>14.1%} {f - r:>+8.1%}")

    print("\n  A real article cannot have a fact-check about it, so the REAL column is pure false")
    print("  match. Any threshold where it is high tells you the ceiling above it is an artefact of")
    print("  comparing one headline against thousands of claims, not evidence of coverage.")

    strict = [t for t in SWEEP
              if sum(1 for s_ in real_scores if s_ >= t) / len(real_scores) <= 0.05]
    if strict:
        t = min(strict)
        ceiling = sum(1 for s_ in fake_scores if s_ >= t) / len(fake_scores)
        fp = sum(1 for s_ in real_scores if s_ >= t) / len(real_scores)
        print(f"\n{'=' * 84}")
        print("Decomposition, at the loosest threshold whose false-match rate stays under 5%")
        print("=" * 84)
        print(f"  threshold                                 {t:.2f} (real false-match {fp:.1%})")
        print(f"  corpus ceiling (a check plausibly exists) {ceiling:6.1%}")
        print("  live API, useful retrieval                  16.0%   (2026-08-24, same sample/seed)")
        if ceiling > 0.16:
            print(f"\n  So coverage is NOT the binding constraint: a check exists for {ceiling:.0%} of these")
            print("  articles and the live search surfaces 16%. The gap is query matching and index")
            print("  differences, both of which are addressable; thin coverage would not have been.")
        elif ceiling > 0:
            print(f"\n  Coverage IS close to binding here: only {ceiling:.0%} have a plausible check in")
            print("  this corpus, against 16% surfaced live. Note the two collections differ, so this")
            print("  understates the true ceiling rather than settling it.")
    else:
        print("\n  No threshold in the sweep gets the false-match rate under 5%. The heuristic cannot")
        print("  support a ceiling estimate on a corpus this size - report that, do not pick a number.")

    pubs = collections.Counter(
        r["publisher"] for r in out_rows
        if r["group"] == "fake" and r["publisher"] and r["score"] >= (min(strict) if strict else 1.1)
    )
    if pubs:
        print("\n  Publishers behind the surviving matches:")
        for name, n in pubs.most_common(8):
            print(f"    {n:>4}  {name or '(unnamed)'}")

    out = _results_dir() / "claimreview_coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
