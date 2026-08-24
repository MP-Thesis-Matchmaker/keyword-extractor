"""Command-line interface: generate benchmark queries, evaluate them.

generate: DB -> clean -> KeyBERT -> aggregate -> NL query -> JSONL.
evaluate: JSONL -> backend-core retriever -> hit@k / MRR report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from keyword_extractor.benchmark import BenchmarkRow, GoldSupervisor, read_jsonl, write_jsonl
from keyword_extractor.cleaning import prepare_text
from keyword_extractor.config import Settings, get_settings
from keyword_extractor.extraction import KeywordExtractor, aggregate_keywords, build_extractor
from keyword_extractor.fetch import (
    PostingRow,
    PublicationRow,
    SupervisorCandidate,
    connect,
    fetch_postings,
    fetch_publications,
    list_supervisor_candidates,
    sample_supervisors,
)
from keyword_extractor.language import detect_language
from keyword_extractor.synthesis import (
    COMPLEXITIES,
    QueryContext,
    QuerySynthesizer,
    build_synthesizer,
)
from keyword_extractor.synthesis.template import select_keywords


def _compose_doc(title: str, body: str | None) -> str:
    """Title + abstract/description, cleaned -- mirrors the composition the
    index embedded (minus the author keywords, which KeyBERT should discover,
    not be fed)."""
    parts = [prepare_text(title or ""), prepare_text(body or "")]
    return "\n".join(p for p in parts if p)


def _pick_synthesizer(settings: Settings, requested: str) -> QuerySynthesizer:
    """The synthesiser the run asked for.

    What each row records is decided by the result, not here: an LLM
    synthesiser that falls back reports "template" per row.
    """
    if requested == "template":
        from keyword_extractor.synthesis import TemplateSynthesizer

        return TemplateSynthesizer()
    if requested == "llm":
        if not settings.llm_base_url:
            raise SystemExit("--query-style llm requires LLM_BASE_URL to be set")
        from keyword_extractor.synthesis.llm import LLMSynthesizer

        return LLMSynthesizer(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            reasoning_effort=settings.llm_reasoning_effort,
        )
    return build_synthesizer(settings)


def _resolve_cris_uuid(cand: SupervisorCandidate, pubs: list[PublicationRow]) -> str | None:
    if cand.cris_uuid:
        return cand.cris_uuid
    for pub in pubs:
        uuid = pub.author_authority_map.get(cand.name)
        if uuid:
            return uuid
    return None


def _most_common(values: list[str | None]) -> str | None:
    counts = Counter(v for v in values if v)
    return counts.most_common(1)[0][0] if counts else None


def _supervisor_rows(
    conn,
    extractor: KeywordExtractor,
    synthesizer: QuerySynthesizer,
    settings: Settings,
    args: argparse.Namespace,
) -> list[BenchmarkRow]:
    if args.supervisors:
        chosen = [
            SupervisorCandidate(name=name, cris_uuid=None, publication_count=0)
            for name in args.supervisors
        ]
    else:
        candidates = list_supervisor_candidates(
            conn, min_publications=args.min_pubs, language=args.language
        )
        chosen = sample_supervisors(candidates, args.sample, seed=args.seed)
        print(f"supervisor candidates: {len(candidates)}, sampled: {len(chosen)}")

    # One flat extract() call over every document, then sliced back per
    # supervisor: the extractor batches embeddings across the whole run.
    # (publication, doc) pairs rather than two parallel lists, because the row's
    # language and its source_ids both have to follow the documents that
    # actually carried text.
    all_docs: list[str] = []
    slices: list[tuple[SupervisorCandidate, list[tuple[PublicationRow, str]], int]] = []
    for cand in chosen:
        pubs = fetch_publications(
            conn,
            cand.name,
            language=args.language,
            max_pubs=args.max_pubs or None,
        )
        pairs = [(p, doc) for p in pubs if (doc := _compose_doc(p.title, p.abstract))]
        if not pairs:
            print(f"skipping {cand.name!r}: no usable publications", file=sys.stderr)
            continue
        slices.append((cand, pairs, len(all_docs)))
        all_docs.extend(doc for _, doc in pairs)

    per_doc = extractor.extract(all_docs)
    rows: list[BenchmarkRow] = []
    for cand, pairs, start in slices:
        keywords = aggregate_keywords(
            per_doc[start : start + len(pairs)], top_n=settings.keybert_top_n
        )
        if not keywords:
            continue
        pubs = [p for p, _ in pairs]
        cris_uuid = _resolve_cris_uuid(cand, pubs)
        department = _most_common([p.department for p in pubs])
        years = [p.year for p in pubs if p.year]
        # Detected from the text, not read from publication.language: that
        # column describes the record, and 87 of 300 deu-labelled publications
        # carry an English abstract. Both are recorded so the gap stays visible.
        text_language = _most_common([detect_language(doc) for _, doc in pairs])
        # One extraction, three queries: the levels slice the same score-ordered
        # keyword list, so the extra rows cost nothing but synthesis.
        for complexity in COMPLEXITIES:
            context = QueryContext(
                seed=cris_uuid or cand.name,
                department=department,
                language=text_language,
                complexity=complexity,
            )
            result = synthesizer.synthesize(keywords, context)
            rows.append(
                BenchmarkRow(
                    query=result.text,
                    keywords=select_keywords(keywords, complexity),
                    source_type="publication",
                    source_ids=[p.id for p in pubs],
                    gold_supervisors=[GoldSupervisor(name=cand.name, cris_uuid=cris_uuid)],
                    granularity="supervisor",
                    query_style=result.style,
                    complexity=complexity,
                    metadata={
                        "department": department,
                        "year_min": min(years) if years else None,
                        "year_max": max(years) if years else None,
                        "text_language": text_language,
                        "db_language": _most_common([p.language for p in pubs]),
                        "keybert_model": settings.keybert_model,
                        "all_keywords": keywords,
                        "seed": args.seed,
                    },
                )
            )
    return rows


def _posting_rows(
    conn,
    extractor: KeywordExtractor,
    synthesizer: QuerySynthesizer,
    settings: Settings,
    args: argparse.Namespace,
) -> list[BenchmarkRow]:
    postings = fetch_postings(conn, require_supervisor=True)
    usable: list[tuple[PostingRow, str]] = []
    for posting in postings:
        doc = _compose_doc(posting.title, posting.description)
        if doc:
            usable.append((posting, doc))
    per_doc = extractor.extract([doc for _, doc in usable])
    rows: list[BenchmarkRow] = []
    for (posting, _), doc_keywords in zip(usable, per_doc, strict=True):
        keywords = aggregate_keywords([doc_keywords], top_n=settings.keybert_top_n)
        if not keywords:
            continue
        # posting.language is NULL for every posting in the corpus, so detection
        # is the only signal available here -- not a fallback.
        text_language = detect_language(_compose_doc(posting.title, posting.description))
        for complexity in COMPLEXITIES:
            context = QueryContext(
                seed=posting.id,
                department=posting.department,
                language=text_language,
                complexity=complexity,
            )
            result = synthesizer.synthesize(keywords, context)
            rows.append(
                BenchmarkRow(
                    query=result.text,
                    keywords=select_keywords(keywords, complexity),
                    source_type="posting",
                    source_ids=[posting.id],
                    gold_supervisors=[GoldSupervisor(name=s.name) for s in posting.supervisors],
                    granularity="posting",
                    query_style=result.style,
                    complexity=complexity,
                    metadata={
                        "department": posting.department,
                        "faculty": posting.faculty,
                        "text_language": text_language,
                        "db_language": posting.language,
                        "keybert_model": settings.keybert_model,
                        "all_keywords": keywords,
                        "seed": args.seed,
                    },
                )
            )
    return rows


def _generate(args: argparse.Namespace) -> int:
    # "all" and "" are spellings of "no language filter"; the keyword model is
    # multilingual, so restricting the corpus is now the special case.
    if args.language in ("all", ""):
        args.language = None
    settings = get_settings()
    if args.top_n:
        settings = settings.model_copy(update={"keybert_top_n": args.top_n})
    extractor = build_extractor(settings)
    synthesizer = _pick_synthesizer(settings, args.query_style)
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    unknown = sources - {"publications", "postings"}
    if unknown:
        raise SystemExit(f"unknown sources: {', '.join(sorted(unknown))}")

    rows: list[BenchmarkRow] = []
    with connect(settings) as conn:
        if "publications" in sources:
            rows.extend(_supervisor_rows(conn, extractor, synthesizer, settings, args))
        if "postings" in sources:
            rows.extend(_posting_rows(conn, extractor, synthesizer, settings, args))

    count = write_jsonl(rows, Path(args.out))
    by_type = Counter(r.source_type for r in rows)
    print(f"wrote {count} benchmark rows to {args.out} ({dict(by_type)})")
    # A degraded run must be visible in the summary, not only in a log line
    # scrolled past 300 rows ago.
    fell_back = sum(1 for r in rows if r.query_style == "template")
    if args.query_style == "llm" and fell_back:
        print(
            f"WARNING: {fell_back} of {count} rows fell back to template phrasing; "
            "check the logged endpoint error above",
            file=sys.stderr,
        )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from keyword_extractor.evaluate import evaluate

    rows = read_jsonl(Path(args.benchmark))
    if not rows:
        raise SystemExit(f"no rows in {args.benchmark}")
    try:
        report = evaluate(rows, top_k=args.top_k)
    finally:
        # backend-core keeps module-level connection pools whose worker threads
        # otherwise whine at interpreter shutdown; close them when present.
        try:
            from thesis_matchmaker.db import close_pools

            close_pools()
        except ImportError:
            pass
    print(json.dumps(report.summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in report.rows:
                fh.write(json.dumps(row) + "\n")
        print(f"per-row results written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="keyword-extractor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate",
        help="synthesise benchmark queries from the database",
        description="Every supervisor and posting yields three rows -- low, medium and high "
        "complexity -- from one keyword extraction, so row counts are three times the number "
        "of sources (100 supervisors produce 300 rows). Scope a run with --sources and --sample.",
    )
    gen.add_argument("--out", default="data/benchmark.jsonl")
    gen.add_argument("--sources", default="publications,postings")
    gen.add_argument("--supervisors", nargs="*", help="explicit supervisor names; skips sampling")
    gen.add_argument("--sample", type=int, default=25, help="supervisors to sample; 0 = all")
    gen.add_argument("--min-pubs", type=int, default=3)
    gen.add_argument(
        "--max-pubs", type=int, default=20, help="most recent pubs per supervisor; 0 = all"
    )
    gen.add_argument(
        "--language",
        default=None,
        help="restrict publications to one ISO 639-3 language, e.g. eng or deu; "
        "default is all languages",
    )
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--top-n", type=int, default=None, help="override KEYBERT_TOP_N")
    gen.add_argument("--query-style", choices=("auto", "template", "llm"), default="auto")

    ev = sub.add_parser("evaluate", help="trial retrieval: hit@k / MRR for the gold supervisors")
    ev.add_argument("--benchmark", default="data/benchmark.jsonl")
    ev.add_argument("--top-k", type=int, default=10)
    ev.add_argument("--out", default=None, help="optional per-row results JSONL")

    args = parser.parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
