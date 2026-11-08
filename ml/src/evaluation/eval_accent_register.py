"""Does accent folding in the claim index help, and for whom?"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, "/app")

from app.api import db_factcheck, local_factcheck as lf  # noqa: E402
from app.api.factcheck import build_query  # noqa: E402


def strip_accents(text: str) -> str:
    """What a phone-typing user does to a claim, applied deliberately."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(c)
    )


def tokens_accent_sensitive(text: str) -> set[str]:
    """The tokeniser as it stood before 2026-11-07."""
    return {
        t
        for t in lf._WORD_RE.findall((text or "").lower())
        if len(t) > 3 and t not in lf._STOPWORDS_ACCENTED
    }


def _with_tokeniser(tok):
    """Swap the module-level tokeniser for the duration of a build or a search."""

    class _Ctx:
        def __enter__(self):
            self.orig = lf._tokens
            lf._tokens = tok
            return tok

        def __exit__(self, *exc):
            lf._tokens = self.orig
            return False

    return _Ctx()


def load_sample(path: Path, n_per_class: int, seed: int) -> tuple[list[str], list[str]]:
    """The same sample eval_factcheck_retrieval.py draws, so the two are comparable."""
    fakes: list[str] = []
    reals: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            (fakes if int(row["label"]) == 1 else reals).append(text)
    rng = random.Random(seed)
    return (
        rng.sample(fakes, min(n_per_class, len(fakes))),
        rng.sample(reals, min(n_per_class, len(reals))),
    )


def found_rate(texts: list[str], index, tok, deaccent: bool, limit: int) -> float:
    hits = 0
    for text in texts:
        query = build_query(text)
        if deaccent:
            query = strip_accents(query)
        with _with_tokeniser(tok):
            if index.search(query, limit):
                hits += 1
    return 100.0 * hits / len(texts) if texts else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="/app/data/processed/fakerecogna2.csv")
    p.add_argument("--sample", type=int, default=200, help="Texts per class (default 200).")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rows = db_factcheck._fetch_rows()
    if not rows:
        print("[accent-eval] the harvested cache is empty; nothing to measure.")
        return 1

    with _with_tokeniser(tokens_accent_sensitive):
        index_sensitive = lf.LocalClaimIndex(rows=rows)
    index_folded = lf.LocalClaimIndex(rows=rows)
    print(
        f"[accent-eval] {len(rows)} rows -> indexed "
        f"accent-sensitive={len(index_sensitive.rows)} folded={len(index_folded.rows)}"
    )

    fakes, reals = load_sample(Path(args.data), args.sample, args.seed)
    print(f"[accent-eval] {len(fakes)} fakes (positives), {len(reals)} reals (controls)")

    matchers = (
        ("accent-sensitive", index_sensitive, tokens_accent_sensitive),
        ("folded", index_folded, lf._tokens),
    )
    out_rows = []
    print()
    print(f"{'query register':<26}{'matcher':<20}{'fake found':>12}{'false assoc':>14}")
    for register, deaccent in (("as-written", False), ("accents-dropped", True)):
        for name, index, tok in matchers:
            found = found_rate(fakes, index, tok, deaccent, args.limit)
            control = found_rate(reals, index, tok, deaccent, args.limit)
            print(f"{register:<26}{name:<20}{found:>11.1f}%{control:>13.1f}%")
            out_rows.append(
                {
                    "query_register": register,
                    "matcher": name,
                    "positives_rate": round(found / 100.0, 4),
                    "controls_rate": round(control / 100.0, 4),
                    "positives_n": len(fakes),
                    "controls_n": len(reals),
                    "index_rows": len(index.rows),
                }
            )

    results_dir = Path(os.environ.get("RESULTS_DIR", "ml/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "accent-register.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
