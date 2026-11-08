"""Measure the evidence layer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/app")

from app.api.factcheck import (  # noqa: E402
    _STOPWORDS,
    _fetch,
    _keyword_query,
    _map_claims,
    build_checker_status,
    build_query,
    is_enabled,
)
from app.api import local_factcheck  # noqa: E402

sys.path.insert(0, "/app/ml_src/evaluation")
try:
    from llm_client import LLMClient, load_llm_config  # noqa: E402
    from llm_client import is_enabled as llm_enabled  # noqa: E402
except ImportError:
    LLMClient = None  # type: ignore[assignment]

    def llm_enabled() -> bool:
        return False

    def load_llm_config() -> dict:
        return {}

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÿ]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def strategy_production(rec: dict, _ctx: dict) -> str:
    """Whatever production currently does."""
    return build_query(rec["text"])


def strategy_first8(rec: dict, _ctx: dict) -> str:
    """The pre-2026-08-24 keyword bag, kept so the switch stays falsifiable and the fallback path is still measured."""
    return _keyword_query(rec["text"])


def strategy_first_sentence(rec: dict, _ctx: dict) -> str:
    """The lede."""
    first = _SENTENCE_END.split(rec["text"].strip(), maxsplit=1)[0]
    return " ".join(first.split())[:200]


_CHECKER_MARKER_RE = re.compile(
    r"(#\s*(?:boato|fake|fato|checagem|verificado)\b"
    r"|\bé\s+(?:falso|fake|enganoso)\b"
    r"|\b(?:boato|enganoso|desmentido|verificamos)\b)",
    re.IGNORECASE,
)


def strategy_title(rec: dict, _ctx: dict) -> str:
    """The real headline, recovered from the corpus metadata, with checker markers removed."""
    title = _CHECKER_MARKER_RE.sub(" ", (rec.get("title") or "")).strip(" -–|:;,.")
    title = " ".join(title.split())
    return title[:200] if len(title) >= 15 else build_query(rec["text"])


_CLAIM_SYSTEM = (
    "Escreva a consulta de busca ideal para encontrar uma checagem de fatos sobre este texto: "
    "uma frase curta com a alegação central, nomes próprios e números relevantes. "
    "Responda apenas com a frase."
)


def strategy_llm_claim(rec: dict, ctx: dict) -> str:
    """Ask the LLM for the checkable claim, and query the index with that."""
    client = ctx.get("llm")
    if client is None:
        return build_query(rec["text"])
    text, err, _ = client.complete(
        _CLAIM_SYSTEM, rec["text"][:4000], meta={"row": rec.get("_i")}, max_tokens=80
    )
    reason = None
    claim = ""
    if err:
        reason = err[:80]
    elif not text:
        reason = "empty completion"
    else:
        claim = " ".join(text.split()).strip('"\' ')
        if claim.upper().startswith("NENHUMA"):
            reason = "model reported no checkable claim"
        elif len(claim) < 15:
            reason = f"too short: {claim!r}"
    if reason:
        ctx.setdefault("_fallbacks", []).append(reason)
        return build_query(rec["text"])
    return claim[:200]


def strategy_tfidf8(rec: dict, ctx: dict) -> str:
    """The 8 most distinctive terms by TF-IDF, ignoring position."""
    vec, feature_names = ctx["vectoriser"], ctx["feature_names"]
    row = vec.transform([rec["text"]])
    if row.indices.size == 0:
        return build_query(rec["text"])
    order = row.data.argsort()[::-1][:8]
    return " ".join(str(feature_names[row.indices[i]]) for i in order)[:200]


def strategy_full_text(rec: dict, _ctx: dict) -> str:
    """The whole article, for the LOCAL source only."""
    return rec["text"]


STRATEGIES = {
    "llm_claim": strategy_llm_claim,
    "full_text": strategy_full_text,
    "production": strategy_production,
    "first8": strategy_first8,
    "first_sentence": strategy_first_sentence,
    "title": strategy_title,
    "tfidf8": strategy_tfidf8,
}


def _tokens(s: str) -> set[str]:
    return {t for t in _WORD_RE.findall(s.lower()) if len(t) > 3 and t not in _STOPWORDS}


MIN_SHARED_TOKENS = 4


def claim_matches(rec: dict, results: list, threshold: float = 0.30) -> bool:
    """Does any returned check plausibly concern THIS claim?"""
    ref = _tokens(rec.get("title") or rec["text"][:300])
    if not ref:
        return False
    for fc in results:
        other = _tokens(fc.claim or "")
        if not other:
            continue
        shared = len(ref & other)
        if shared >= MIN_SHARED_TOKENS and shared / min(len(ref), len(other)) >= threshold:
            return True
    return False


def load_sample(path: Path, n_per_class: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Return (fake_records, real_records) as {text, title}."""
    fakes, reals = [], []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            rec = {"text": text, "title": (row.get("title") or "").strip()}
            (fakes if int(row["label"]) == 1 else reals).append(rec)
    rng = random.Random(seed)
    return rng.sample(fakes, min(n_per_class, len(fakes))), rng.sample(
        reals, min(n_per_class, len(reals))
    )


def build_context(records: list[dict]) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        lowercase=True, stop_words=sorted(_STOPWORDS), ngram_range=(1, 1),
        min_df=2, max_features=50_000, sublinear_tf=True,
    )
    vec.fit([r["text"] for r in records])
    return {"vectoriser": vec, "feature_names": vec.get_feature_names_out()}


def probe(query: str, limit: int, source: str = "live") -> tuple[list, str | None]:
    """One retrieval call against one source."""
    if not query:
        return [], "empty query"
    if source == "local":
        try:
            return local_factcheck.search_local(query, limit), None
        except Exception as exc:  # noqa: BLE001
            return [], f"local: {type(exc).__name__}: {exc}"
    if source == "db":
        try:
            from app.api import db_factcheck

            return db_factcheck.search_db(query, limit), None
        except Exception as exc:  # noqa: BLE001
            return [], f"db: {type(exc).__name__}: {exc}"
    try:
        return _map_claims(_fetch(query), limit), None
    except Exception as exc:  # noqa: BLE001 - counted and reported, never hidden
        return [], f"{type(exc).__name__}: {exc}"


def arm_fingerprint(raw_rows: list[dict], strategy: str) -> str:
    """A hash of everything one arm actually retrieved."""
    items = sorted(
        (r["group"], r.get("title", ""), r["url"])
        for r in raw_rows
        if r["strategy"] == strategy
    )
    return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()[:12]


def warn_identical_arms(raw_rows: list[dict], strategies: list[str]) -> list[tuple[str, str]]:
    """Report any pair of arms that retrieved exactly the same things."""
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []
    for name in strategies:
        fp = arm_fingerprint(raw_rows, name)
        if fp in seen:
            collisions.append((seen[fp], name))
        else:
            seen[fp] = name
    for first, second in collisions:
        print(f"\n  !! IDENTICAL RESULTS: '{first}' and '{second}' retrieved exactly the same items.")
        print("     Verify they are exercising different code paths before reading this as a null")
        print("     result. Three past incidents with this signature were all wiring faults.")
    return collisions


def dedupe(results: list) -> list:
    """Union of results from several sources, de-duplicated by URL."""
    seen, out = set(), []
    for fc in results:
        key = (fc.url or fc.claim or "").lower()
        if key and key not in seen:
            seen.add(key)
            out.append(fc)
    return out


def run(args: argparse.Namespace) -> int:
    data = Path(args.data)
    if not data.exists():
        print(f"[factcheck-eval] ERROR: {data} not found.")
        return 1

    fakes, reals = load_sample(data, args.sample, args.seed)
    print(f"[factcheck-eval] {data.name}: {len(fakes)} fakes (positives), {len(reals)} reals (controls)")

    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STRATEGIES]
    if unknown:
        print(f"[factcheck-eval] ERROR: unknown strategies {unknown}; known: {list(STRATEGIES)}")
        return 1

    sources = [x.strip() for x in args.sources.split(",") if x.strip()]
    bad = [x for x in sources if x not in ("live", "local", "db")]
    if bad:
        print(f"[factcheck-eval] ERROR: unknown source(s) {bad}; known: live, local, db")
        return 1
    if "db" in sources:
        from app.api import db_factcheck as _db

        if not _db.is_enabled():
            print(f"[factcheck-eval] the db source is unavailable: {_db.status()}")
            return 1
    if "local" in sources and not local_factcheck.is_enabled():
        print(f"[factcheck-eval] ERROR: local corpus unavailable "
              f"({local_factcheck.get_index().error}). Build it with prepare_claimreview_feed.py.")
        return 1

    ctx = build_context(fakes + reals) if "tfidf8" in wanted else {}
    if "llm_claim" in wanted:
        if LLMClient is None or not llm_enabled():
            print("[factcheck-eval] ERROR: llm_claim needs MARITACA_API_KEY and llm_client.py.")
            return 1
        ctx["llm"] = LLMClient(cfg=load_llm_config(), run_name="claim_extraction")
    total_calls = (len(fakes) + len(reals)) * len(wanted) * max(1, len([s for s in
                  [x.strip() for x in args.sources.split(",")] if s == "live"]))

    if args.dry_run:
        print(f"\n[dry-run] would make {total_calls} API calls "
              f"({len(fakes) + len(reals)} texts x {len(wanted)} strategies). Example queries:\n")
        for name in wanted:
            print(f"  --- {name} ---")
            if name == "llm_claim":
                print(f"    [would send this system prompt, max_tokens=80]")
                print(f"    {_CLAIM_SYSTEM[:150]}")
                print(f"    [with the article text, e.g.] {fakes[0]['text'][:90]}...")
                continue
            for rec in fakes[:2]:
                print(f"    {STRATEGIES[name](rec, ctx)[:110]}")
        print("\n[dry-run] no calls made. Drop --dry-run to run for real.")
        return 0

    if "live" in sources and not is_enabled():
        print("\n[factcheck-eval] GOOGLE_FACTCHECK_API_KEY is not set, so the LIVE source is "
              "DISABLED and no external call was made.\n[factcheck-eval] This is the offline "
              "default, not a result. Set the key in .env, or pass --sources local to measure the "
              "offline corpus on its own.")
        return 0

    print(f"[factcheck-eval] {total_calls} API calls, {args.delay}s apart")
    print(f"[factcheck-eval] sources: {', '.join(sources)}"
          + (f" | local corpus: {len(local_factcheck.get_index().rows)} checks"
             if "local" in sources else "") + "\n")
    summary, publishers = [], Counter()
    raw_rows: list[dict] = []
    recognised, unmatched = Counter(), Counter()

    for name in wanted:
        fn = STRATEGIES[name]
        row = {"strategy": name}
        source_hits: Counter = Counter()
        for group, records in (("positives", fakes), ("controls", reals)):
            hits = errors = matched = 0
            for rec in records:
                query = fn(rec, ctx)
                offline_query = strategy_production(rec, ctx) if args.per_source_query else query
                per_source, err = {}, None
                for src in sources:
                    r, e = probe(offline_query if src in ("local", "db") else query,
                                 args.max_results, source=src)
                    per_source[src] = r
                    if e and src == "live":
                        err = e
                    source_hits[src] += bool(r)
                results = dedupe([fc for r in per_source.values() for fc in r])
                for fc in results:
                    raw_rows.append({
                        "strategy": name, "group": group, "query": query,
                        "title": rec.get("title", ""), "claim": fc.claim,
                        "rating": fc.rating, "publisher": fc.publisher, "url": fc.url,
                    })
                if err:
                    errors += 1
                elif results:
                    hits += 1
                    if claim_matches(rec, results):
                        matched += 1
                    if group == "positives":
                        publishers.update(fc.publisher for fc in results)
                        recognised.update(
                            c.name for c in build_checker_status(results) if c.found
                        )
                        unmatched.update(
                            fc.publisher for fc in results
                            if not any(c.found for c in build_checker_status([fc]))
                        )
                time.sleep(args.delay)
            ok = len(records) - errors
            row[f"{group}_rate"] = hits / ok if ok else 0.0
            row[f"{group}_n"] = ok
            row[f"{group}_errors"] = errors
            row[f"{group}_matched"] = matched
            row[f"{group}_precision"] = matched / hits if hits else 0.0
            row[f"{group}_useful_rate"] = matched / ok if ok else 0.0
        fb = ctx.pop("_fallbacks", [])
        if fb:
            from collections import Counter as _C
            print(f"  [{name}] {len(fb)} fell back to the production query: "
                  f"{dict(_C(r.split(':')[0] for r in fb).most_common(3))}")
        n_pos = max(1, row.get("positives_n", 1))
        row["by_source"] = " ".join(
            f"{src}:{source_hits[src] / (2 * n_pos):.1%}" for src in sources
        )
        for src in sources:
            if source_hits[src] == 0:
                print(f"  !! SOURCE '{src}' RETURNED NOTHING across all {2 * n_pos} queries in "
                      f"'{name}'.")
                print("     Either the index is empty or the source is not wired. Do not read the "
                      "numbers below as a measurement of it.")
        summary.append(row)
        print(f"  {name:<16} positives found {row['positives_rate']:6.1%} "
              f"-> on-claim {row['positives_precision']:6.1%} "
              f"= useful {row['positives_useful_rate']:6.1%}   "
              f"controls found {row['controls_rate']:6.1%} "
              f"(n={row['positives_n']}, err={row['positives_errors'] + row['controls_errors']})")

    print("\n" + "=" * 78)
    print("Reading")
    print("=" * 78)
    print("  positives = fake articles for which a published check demonstrably exists, where")
    print("              retrieval surfaced SOMETHING related. An upper bound on useful recall.")
    print("  controls  = legitimate articles that nonetheless matched a check. In the UI this puts")
    print("              'checagem relacionada' beside real journalism, so it is a false-association")
    print("              rate, not a neutral number.")
    print("  on-claim  = of the hits, the share where a returned claim shares enough vocabulary with")
    print("              the article's headline to plausibly be about it. A similarity heuristic, not")
    print("              adjudication - but it separates 'found something about this article' from")
    print("              'found something about Brazilian politics'.")
    print("  useful    = found AND on-claim. This is the honest end-to-end retrieval rate.")
    print("  Differences BETWEEN strategies are attributable to query construction alone - the index")
    print("  is identical across rows. A flat comparison points at index coverage instead.")

    if publishers:
        print("\n  Publishers returned for positives (top 12):")
        for pub, count in publishers.most_common(12):
            print(f"    {count:>5}  {pub}")
        print("\n  Monitored checkers the per-checker panel would light up "
              "(via build_checker_status, i.e. URL-domain matching as in production):")
        for name, count in recognised.most_common():
            print(f"    {count:>5}  {name}")
        if unmatched:
            print("\n  Publishers returned that match NO monitored checker - the panel stays silent")
            print("  about these even though a relevant check was found:")
            for pub, count in unmatched.most_common(10):
                print(f"    {count:>5}  {pub}")

    if len(wanted) > 1:
        warn_identical_arms(raw_rows, wanted)

    results_dir = Path(os.environ.get("RESULTS_DIR", "/app/results"))
    suffix = "" if set(sources) == {"live", "local"} else f"-{'-'.join(sorted(sources))}"
    out = results_dir / f"factcheck-retrieval{suffix}.csv"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(summary)
        print(f"\n  wrote {out}")
        if raw_rows:
            raw_out = results_dir / f"factcheck-retrieval{suffix}-raw.csv"
            with open(raw_out, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(raw_rows[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(raw_rows)
            print(f"  wrote {raw_out}  ({len(raw_rows)} returned claims - re-analysis needs no API)")
    except OSError as exc:
        print(f"\n  could not write results ({exc}); the table above is the record")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measure ClaimReview retrieval recall and precision.")
    p.add_argument("--data", default="/app/data/processed/fakerecogna2.csv")
    p.add_argument("--sample", type=int, default=200, help="Texts per class (default 200).")
    p.add_argument("--strategies", default="production")
    p.add_argument("--per-source-query", action="store_true", default=True,
                   help="Pin the OFFLINE sources (local, db) to the production query while a "
                        "different strategy is tested against the live API (default on). This is a "
                        "control, so a live-side change cannot move the offline arms - but it also "
                        "means comparing strategies with --sources local is meaningless, because "
                        "every arm then runs the same query. Pass --no-per-source-query to compare "
                        "strategies offline. Guard A now says so out loud when it happens.")
    p.add_argument("--no-per-source-query", dest="per_source_query", action="store_false")
    p.add_argument("--sources", default="live",
                   help="Comma-separated retrieval sources: live (Google Claim Search), local "
                        "(the bundled ClaimReview DataFeed corpus), db (the harvested Postgres "
                        "cache), or any combination. They hold different publishers and different "
                        "eras, so the union is larger than any one of them.")
    p.add_argument("--max-results", type=int, default=5)
    p.add_argument("--delay", type=float, default=0.15, help="Seconds between calls.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Print queries and call count; call nothing.")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
