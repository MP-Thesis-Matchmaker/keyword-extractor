"""Command-line interface: generate benchmark queries, evaluate them.

generate: DB -> clean -> KeyBERT -> aggregate -> NL query -> JSONL.
evaluate: JSONL -> backend-core retriever -> hit@k / MRR report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from keyword_extractor.benchmark import (
    BenchmarkManifest,
    BenchmarkRow,
    GoldSupervisor,
    manifest_path,
    read_jsonl,
    read_manifest,
    utc_now,
    write_jsonl,
    write_manifest,
)
from keyword_extractor.cleaning import prepare_text
from keyword_extractor.config import Settings, get_settings
from keyword_extractor.extraction import KeywordExtractor, aggregate_keywords, build_extractor
from keyword_extractor.fetch import (
    CRIS,
    PostingRow,
    PublicationRow,
    SupervisorCandidate,
    connect,
    corpus_stats,
    cris_uuid_for,
    fetch_postings,
    fetch_publications,
    list_supervisor_candidates,
    sample_postings,
    sample_supervisors,
    schema_fingerprint,
)
from keyword_extractor.language import detect_language
from keyword_extractor.matcher_client import MatcherClient, MatcherError
from keyword_extractor.synthesis import (
    COMPLEXITIES,
    QueryContext,
    QuerySynthesizer,
    build_synthesizer,
)
from keyword_extractor.synthesis.template import select_keywords

# Old name -> new name. The old set read as if the three counts scoped the
# same thing, while --sample counted supervisors, --min-pubs decided which
# supervisors qualified and --max-pubs capped their documents.
_RENAMED_FLAGS = {
    "--sample": "--supervisor-sample",
    "--min-pubs": "--supervisor-min-pubs",
    "--max-pubs": "--supervisor-max-pubs",
    "--supervisors": "--supervisor-names",
    "--language": "--publication-language",
    "--top-n": "--keyword-top-n",
}


class _RenamedFlag(argparse.Action):
    """Exit naming the replacement, rather than "unrecognized arguments".

    The old names are gone rather than aliased -- keeping both spellings alive
    is what lets an ambiguous one stay in circulation -- but a line recalled
    from shell history deserves an answer that says where the flag went.
    """

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: ARG002
        raise SystemExit(
            f"{option_string} was renamed to {_RENAMED_FLAGS[option_string]}; see --help"
        )


def _add_renamed_flags(parser: argparse.ArgumentParser) -> None:
    for old in _RENAMED_FLAGS:
        parser.add_argument(
            old,
            action=_RenamedFlag,
            nargs="*",
            dest="_renamed",
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )


def _add_log_level(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
        help="info shows endpoint retries and field drops; warning hides them",
    )


def _configure_logging(level: str) -> None:
    """Route our own log lines to stderr and keep httpx from burying them.

    Without this the module loggers sit below the root logger's default
    WARNING threshold, so the endpoint's own explanation of a failure was
    discarded and only the fallback warning ever reached the terminal.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # One "HTTP Request: POST ... 200 OK" per row would drown the progress counter.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    if total >= 3600:
        return f"{total // 3600}h{total % 3600 // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total}s"


class _Progress:
    """Row counter with an ETA, so a long run reads as working rather than hung.

    Every `every` rows, not every row: an LLM run over the full posting corpus
    is ~1700 sequential calls across hours and prints about 70 lines, while a
    three-row smoke run prints one. stderr, so `--out` piping stays clean.
    """

    def __init__(self, label: str, total: int, every: int = 25) -> None:
        self.label = label
        self.total = total
        self.every = every
        self.done = 0
        self._start = time.monotonic()

    def advance(self) -> None:
        self.done += 1
        if self.done % self.every == 0 and self.done != self.total:
            self._emit()

    def finish(self) -> None:
        if self.total:
            self._emit()

    def _emit(self) -> None:
        line = f"{self.label:<13}{self.done}/{self.total} rows"
        if 0 < self.done < self.total:
            elapsed = time.monotonic() - self._start
            line += f"  eta {_format_duration(elapsed / self.done * (self.total - self.done))}"
        print(line, file=sys.stderr, flush=True)


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
            max_attempts=settings.llm_max_attempts,
        )
    return build_synthesizer(settings)


def _resolve_cris_uuid(cand: SupervisorCandidate, pubs: list[PublicationRow]) -> str | None:
    """The supervisor\'s CRIS Person UUID, from the candidate row or any publication.

    `cris_uuid_for` rather than a plain map lookup: an orcid-typed authority also
    carries an id, and writing that into a field named cris_uuid is how a gold
    label starts claiming a Person that does not exist.
    """
    if cand.cris_uuid:
        return cand.cris_uuid
    for pub in pubs:
        uuid = cris_uuid_for(pub.author_authority_map, cand.name)
        if uuid:
            return uuid
    return None


def build_matcher_client(settings: Settings, *, base_url: str | None = None) -> MatcherClient:
    """Constructed through a module-level function so tests can substitute a stub."""
    return MatcherClient(settings, base_url=base_url)


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
    if args.supervisor_names:
        chosen = [
            SupervisorCandidate(name=name, cris_uuid=None, publication_count=0)
            for name in args.supervisor_names
        ]
    else:
        pool = list_supervisor_candidates(
            conn,
            min_publications=args.supervisor_min_pubs,
            language=args.publication_language,
            require_cris_uuid=args.require_cris_uuid,
        )
        chosen = sample_supervisors(pool.candidates, args.supervisor_sample, seed=args.seed)
        # stderr, with the progress counter: it is status, and mixing the two
        # streams reorders them the moment the run is piped to a file.
        # `cris-typed` below `candidates` is the eligibility invariant: equal
        # numbers mean upstream still admits only registered UZH researchers.
        print(
            f"supervisor candidates: {pool.total}, cris-typed: {pool.cris_typed}, "
            f"sampled: {len(chosen)}",
            file=sys.stderr,
            flush=True,
        )

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
            language=args.publication_language,
            max_pubs=args.supervisor_max_pubs or None,
        )
        pairs = [(p, doc) for p in pubs if (doc := _compose_doc(p.title, p.abstract))]
        if not pairs:
            print(f"skipping {cand.name!r}: no usable publications", file=sys.stderr)
            continue
        slices.append((cand, pairs, len(all_docs)))
        all_docs.extend(doc for _, doc in pairs)

    per_doc = extractor.extract(all_docs)
    rows: list[BenchmarkRow] = []
    progress = _Progress("publications", len(slices) * len(COMPLEXITIES))
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
            # Gold publication count: how many indexed publications the
            # supervisor has. cand.publication_count is from the DB query
            # (accurate); len(pubs) is the fallback for --supervisor-names
            # (capped at --supervisor-max-pubs, but still usable).
            gold_pub_count = cand.publication_count or len(pubs)
            difficulty = "easy" if gold_pub_count >= args.difficulty_threshold else "hard"
            rows.append(
                BenchmarkRow(
                    query=result.text,
                    keywords=select_keywords(keywords, complexity),
                    source_type="publication",
                    source_ids=[p.id for p in pubs],
                    gold_supervisors=[
                        GoldSupervisor(
                            name=cand.name,
                            cris_uuid=cris_uuid,
                            authority_type=cand.authority_type or (CRIS if cris_uuid else None),
                        )
                    ],
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
                        "gold_pub_count": gold_pub_count,
                        "difficulty": difficulty,
                        "difficulty_threshold": args.difficulty_threshold,
                    },
                )
            )
            progress.advance()
    progress.finish()
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
    # Sample before extracting, not after: a posting is its own benchmark row,
    # so this is what bounds the run. Filtering first keeps the sample at full
    # size instead of losing members to postings that carry no text.
    sampled = sample_postings([p for p, _ in usable], args.posting_sample, seed=args.seed)
    chosen_ids = {p.id for p in sampled}
    usable = [pair for pair in usable if pair[0].id in chosen_ids]
    per_doc = extractor.extract([doc for _, doc in usable])
    rows: list[BenchmarkRow] = []
    progress = _Progress("postings", len(usable) * len(COMPLEXITIES))
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
                        "gold_pub_count": None,
                        "difficulty": "easy",
                        "difficulty_threshold": args.difficulty_threshold,
                    },
                )
            )
            progress.advance()
    progress.finish()
    return rows



_SPECIFICITY_RUNGS = ("general", "adjacent", "true")


def _parse_adjacent_map(adjacent_args: list[str] | None) -> dict[str, str]:
    """Parse --adjacent 'Name=subfield' arguments into a lookup dict."""
    result: dict[str, str] = {}
    for entry in adjacent_args or []:
        if "=" not in entry:
            raise SystemExit(
                f"--adjacent requires 'SupervisorName=adjacent subfield' format, got: {entry!r}"
            )
        name, subfield = entry.split("=", 1)
        result[name.strip()] = subfield.strip()
    return result


def _specificity_rows(
    conn,
    extractor: KeywordExtractor,
    synthesizer: QuerySynthesizer,
    settings: Settings,
    args: argparse.Namespace,
) -> list[BenchmarkRow]:
    """Build specificity-ablation rows: 3 rungs (general/adjacent/true) per supervisor.

    Requires --supervisor-names and --adjacent. The 'true' and 'general' rungs
    are derived from KeyBERT (top-ranked = most specific = true, lowest-ranked
    = broadest = general). The 'adjacent' rung is human-provided.
    """
    if not args.supervisor_names:
        raise SystemExit("--specificity requires --supervisor-names")

    adjacent_map = _parse_adjacent_map(getattr(args, "adjacent", None))

    chosen = [
        SupervisorCandidate(name=name, cris_uuid=None, publication_count=0)
        for name in args.supervisor_names
    ]

    all_docs: list[str] = []
    slices: list[tuple[SupervisorCandidate, list[tuple[PublicationRow, str]], int]] = []
    for cand in chosen:
        if cand.name not in adjacent_map:
            raise SystemExit(
                f"--adjacent missing for supervisor {cand.name!r}. "
                f"Provide: --adjacent \"{cand.name}=<adjacent subfield>\""
            )
        pubs = fetch_publications(
            conn,
            cand.name,
            language=args.publication_language,
            max_pubs=args.supervisor_max_pubs or None,
        )
        pairs = [(p, doc) for p in pubs if (doc := _compose_doc(p.title, p.abstract))]
        if not pairs:
            print(f"skipping {cand.name!r}: no usable publications", file=sys.stderr)
            continue
        slices.append((cand, pairs, len(all_docs)))
        all_docs.extend(doc for _, doc in pairs)

    per_doc = extractor.extract(all_docs)
    rows: list[BenchmarkRow] = []
    progress = _Progress("specificity", len(slices) * 3)

    for cand, pairs, start in slices:
        keywords = aggregate_keywords(
            per_doc[start : start + len(pairs)], top_n=settings.keybert_top_n
        )
        if not keywords:
            continue

        pubs = [p for p, _ in pairs]
        cris_uuid = _resolve_cris_uuid(cand, pubs)
        department = _most_common([p.department for p in pubs])
        text_language = _most_common([detect_language(doc) for _, doc in pairs])

        # true = top keyphrase (most specific to this supervisor)
        true_topic = keywords[0]
        # general = last keyphrase (broadest / least specific)
        general_topic = keywords[-1]
        # adjacent = human-provided subfield they don't work in
        adjacent_topic = adjacent_map[cand.name]

        rung_topics = {
            "true": true_topic,
            "general": general_topic,
            "adjacent": adjacent_topic,
        }

        for rung, topic in rung_topics.items():
            context = QueryContext(
                seed=cris_uuid or cand.name,
                department=department,
                language=text_language,
                complexity="medium",
            )
            result = synthesizer.synthesize([topic], context)
            rows.append(
                BenchmarkRow(
                    query=result.text,
                    keywords=[topic],
                    source_type="publication",
                    source_ids=[p.id for p in pubs],
                    gold_supervisors=[
                        GoldSupervisor(
                            name=cand.name,
                            cris_uuid=cris_uuid,
                            authority_type=cand.authority_type or (CRIS if cris_uuid else None),
                        )
                    ],
                    granularity="supervisor",
                    query_style=result.style,
                    complexity="n/a",
                    metadata={
                        "department": department,
                        "text_language": text_language,
                        "keybert_model": settings.keybert_model,
                        "all_keywords": keywords,
                        "seed": args.seed,
                        "rung": rung,
                        "rung_topic": topic,
                    },
                )
            )
            progress.advance()
    progress.finish()
    return rows




# ---------------------------------------------------------------------------
# Part 3: Student-background ablation
# ---------------------------------------------------------------------------

_BACKGROUND_PROMPT = (
    "Given this thesis topic, invent a plausible one-sentence student background "
    "(major, modules taken, a past project). Do not mention any person or supervisor "
    "names. Return only the sentence."
)

_ABLATION_ARMS = ("topic_only", "plus_program", "plus_background", "full")

# Simple department → faculty mapping for study_program field.
_DEPT_TO_FACULTY: dict[str, str] = {
    "Department of Informatics": "Faculty of Business, Economics and Informatics",
    "Department of Finance": "Faculty of Business, Economics and Informatics",
    "Department of Economics": "Faculty of Business, Economics and Informatics",
    "Department of Business Administration": "Faculty of Business, Economics and Informatics",
    "Department of Political Science": "Faculty of Arts and Social Sciences",
    "Department of Chemistry": "Faculty of Science",
    "Physics Institute": "Faculty of Science",
    "Institute of Geography": "Faculty of Science",
    "Institute of Computational Linguistics": "Faculty of Arts and Social Sciences",
    "Institute of Psychology": "Faculty of Arts and Social Sciences",
    "Institute of Evolutionary Biology and Environmental Studies": "Faculty of Science",
    "Institute of Biomedical Engineering": "Faculty of Medicine and Science",
    "Institute of History": "Faculty of Arts and Social Sciences",
    "Institute of Philosophy": "Faculty of Arts and Social Sciences",
    "Institute of Sociology": "Faculty of Arts and Social Sciences",
    "Institute of Education": "Faculty of Arts and Social Sciences",
}


def _generate_background_text(
    department: str | None,
    keywords: list[str],
    synthesizer: QuerySynthesizer,
    seed: int,
    cache: dict[str, str],
) -> str:
    """Generate a plausible student background sentence.

    Cached by department + seed so the same department always produces the same
    background within a run. Falls back to a deterministic template if the LLM
    is unavailable.
    """
    cache_key = f"{department or 'unknown'}:{seed}"
    if cache_key in cache:
        return cache[cache_key]

    # Try LLM synthesis via a short topic summary
    topic_summary = ", ".join(keywords[:4])
    try:
        from keyword_extractor.synthesis.llm import LLMSynthesizer

        if isinstance(synthesizer, LLMSynthesizer):
            text = synthesizer._chat(
                f"Thesis topic: {topic_summary}\n{_BACKGROUND_PROMPT}"
            )
            if text:
                cache[cache_key] = text
                return text
    except Exception:
        pass

    # Deterministic fallback
    dept_short = (department or "my department").split("of ")[-1].strip() if department else "my field"
    fallback = (
        f"I have completed coursework in {dept_short} and worked on "
        f"a semester project related to {keywords[0] if keywords else 'this topic'}."
    )
    cache[cache_key] = fallback
    return fallback


def _ablation_background_rows(
    conn,
    extractor: KeywordExtractor,
    synthesizer: QuerySynthesizer,
    settings: Settings,
    args: argparse.Namespace,
) -> list[BenchmarkRow]:
    """Build 4-arm ablation rows: topic_only / plus_program / plus_background / full.

    Reuses the normal supervisor sampling and keyword extraction, then for each
    source row produces 4 variants differing only in which student-context
    fields are included in the query.
    """
    # First, generate the normal supervisor rows to get keywords and metadata
    all_docs: list[str] = []
    slices: list[tuple[SupervisorCandidate, list[tuple[PublicationRow, str]], int]] = []

    if args.supervisor_names:
        chosen = [
            SupervisorCandidate(name=name, cris_uuid=None, publication_count=0)
            for name in args.supervisor_names
        ]
    else:
        pool = list_supervisor_candidates(
            conn,
            min_publications=args.supervisor_min_pubs,
            language=args.publication_language,
            require_cris_uuid=args.require_cris_uuid,
        )
        chosen = sample_supervisors(pool.candidates, args.supervisor_sample, seed=args.seed)
        print(
            f"supervisor candidates: {pool.total}, cris-typed: {pool.cris_typed}, "
            f"sampled: {len(chosen)}",
            file=sys.stderr,
            flush=True,
        )

    for cand in chosen:
        pubs = fetch_publications(
            conn,
            cand.name,
            language=args.publication_language,
            max_pubs=args.supervisor_max_pubs or None,
        )
        pairs = [(p, doc) for p in pubs if (doc := _compose_doc(p.title, p.abstract))]
        if not pairs:
            print(f"skipping {cand.name!r}: no usable publications", file=sys.stderr)
            continue
        slices.append((cand, pairs, len(all_docs)))
        all_docs.extend(doc for _, doc in pairs)

    per_doc = extractor.extract(all_docs)
    rows: list[BenchmarkRow] = []
    background_cache: dict[str, str] = {}
    progress = _Progress("background-ablation", len(slices) * len(_ABLATION_ARMS))

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
        text_language = _most_common([detect_language(doc) for _, doc in pairs])

        # Derive student context once per supervisor
        faculty = _DEPT_TO_FACULTY.get(department or "", department or "")
        study_program = f"{faculty}" if faculty else None
        student_background = _generate_background_text(
            department, keywords, synthesizer, args.seed, background_cache
        )

        # 4 arms from the same keyword set, using medium complexity
        arm_configs = {
            "topic_only": (None, None),
            "plus_program": (study_program, None),
            "plus_background": (None, student_background),
            "full": (study_program, student_background),
        }

        for arm, (prog, bg) in arm_configs.items():
            context = QueryContext(
                seed=cris_uuid or cand.name,
                department=department,
                language=text_language,
                complexity="medium",
                study_program=prog,
                student_background=bg,
            )
            result = synthesizer.synthesize(keywords, context)
            rows.append(
                BenchmarkRow(
                    query=result.text,
                    keywords=select_keywords(keywords, "medium"),
                    source_type="publication",
                    source_ids=[p.id for p in pubs],
                    gold_supervisors=[
                        GoldSupervisor(
                            name=cand.name,
                            cris_uuid=cris_uuid,
                            authority_type=cand.authority_type or (CRIS if cris_uuid else None),
                        )
                    ],
                    granularity="supervisor",
                    query_style=f"ablation:{arm}",
                    complexity="medium",
                    metadata={
                        "department": department,
                        "year_min": min(years) if years else None,
                        "year_max": max(years) if years else None,
                        "text_language": text_language,
                        "db_language": _most_common([p.language for p in pubs]),
                        "keybert_model": settings.keybert_model,
                        "all_keywords": keywords,
                        "seed": args.seed,
                        "ablation_arm": arm,
                        "study_program": prog,
                        "student_background": bg,
                        "actual_style": result.style,
                    },
                )
            )
            progress.advance()
    progress.finish()
    return rows


def _warn_on_index_drift(
    manifest: BenchmarkManifest | None, index: dict, benchmark: str
) -> None:
    """Say so when the benchmark was generated against a different index.

    Deliberately not a check against the live database's schema fingerprint:
    evaluation no longer opens a connection, and reintroducing one to validate a
    manifest would undo the point of the HTTP migration. The realistic failure
    is evaluating an old benchmark against a re-indexed matcher, and that is
    visible from the manifest alone.
    """
    if manifest is None:
        print(
            f"NOTE: {benchmark} has no run manifest, so the corpus its gold "
            "labels came from is unrecorded. Regenerate it to get one.",
            file=sys.stderr,
            flush=True,
        )
        return
    was = (manifest.index or {}).get("embedding_model")
    now = index.get("embedding_model")
    if was and now and was != now:
        print(
            f"WARNING: this benchmark was generated against an index built with {was}, "
            f"but the matcher answering is on {now}. The scores are not comparable "
            "with earlier runs of the same file.",
            file=sys.stderr,
            flush=True,
        )


def _index_status_if_reachable(settings: Settings) -> dict | None:
    """The matcher's index status, or None when there is no matcher to ask.

    Generation does not need a matcher, so its absence is not an error. Recording
    the index when one *is* running makes a generate/evaluate pair traceable to
    the same state of the system.
    """
    try:
        with build_matcher_client(settings) as client:
            return client.index_status()
    except MatcherError:
        return None


def _generate(args: argparse.Namespace) -> int:
    # "all" and "" are spellings of "no language filter"; the keyword model is
    # multilingual, so restricting the corpus is now the special case.
    if args.publication_language in ("all", ""):
        args.publication_language = None
    settings = get_settings()
    if args.keyword_top_n:
        settings = settings.model_copy(update={"keybert_top_n": args.keyword_top_n})
    extractor = build_extractor(settings)
    synthesizer = _pick_synthesizer(settings, args.query_style)
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    unknown = sources - {"publications", "postings"}
    if unknown:
        raise SystemExit(f"unknown sources: {', '.join(sorted(unknown))}")

    rows: list[BenchmarkRow] = []
    with connect(settings) as conn:
        if getattr(args, "specificity", False):
            rows.extend(_specificity_rows(conn, extractor, synthesizer, settings, args))
        elif getattr(args, "ablation_background", False):
            rows.extend(_ablation_background_rows(conn, extractor, synthesizer, settings, args))
        else:
            if "publications" in sources:
                rows.extend(_supervisor_rows(conn, extractor, synthesizer, settings, args))
            if "postings" in sources:
                rows.extend(_posting_rows(conn, extractor, synthesizer, settings, args))
        # Read inside the same connection as the rows themselves, so the
        # manifest describes the corpus these labels actually came from.
        corpus = corpus_stats(conn, min_publications=args.supervisor_min_pubs)
        fingerprint = schema_fingerprint(conn)

    out_path = Path(args.out)
    count = write_jsonl(rows, out_path)
    write_manifest(
        BenchmarkManifest(
            generated_at=utc_now(),
            schema_fingerprint=fingerprint,
            corpus=corpus,
            index=_index_status_if_reachable(settings),
            run={
                "argv": sys.argv[1:],
                "rows": count,
                "seed": args.seed,
                "keybert_model": settings.keybert_model,
                "keybert_top_n": settings.keybert_top_n,
                "supervisor_sample": args.supervisor_sample,
                "supervisor_min_pubs": args.supervisor_min_pubs,
                "supervisor_max_pubs": args.supervisor_max_pubs,
                "posting_sample": args.posting_sample,
                "publication_language": args.publication_language,
                "require_cris_uuid": args.require_cris_uuid,
                "query_style": args.query_style,
                "sources": sorted(sources),
            },
        ),
        out_path,
    )
    by_type = Counter(r.source_type for r in rows)
    print(f"wrote {count} benchmark rows to {args.out} ({dict(by_type)})", flush=True)
    print(f"wrote run manifest to {manifest_path(out_path)}", flush=True)
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


# Two separate ceilings, and the lower one is not the documented one.
#
# The documented ceiling is the matcher's psycopg pool (max_size=5, synchronous
# routes), so a fifth concurrent DB-touching request meets PoolTimeout. 4 leaves
# one connection for the preflight.
#
# The real ceiling, measured against the running matcher, is 1. Uvicorn dispatches
# sync `def` routes to a threadpool, and every one of them embeds the query with a
# process-wide SentenceTransformer whose HuggingFace fast tokenizer is Rust-backed
# and not thread-safe: two concurrent queries reconfiguring its truncation and
# padding race, and the loser gets `RuntimeError: Already borrowed` -> 500. That is
# an upstream bug, not something a client can work around, which is why the default
# is 1 and why --concurrency carries a warning rather than being removed.
_MAX_CONCURRENCY = 4


def _evaluate(args: argparse.Namespace) -> int:
    from keyword_extractor.evaluate import evaluate

    rows = read_jsonl(Path(args.benchmark))
    if not rows:
        raise SystemExit(f"no rows in {args.benchmark}")

    settings = get_settings()
    concurrency = args.concurrency if args.concurrency is not None else settings.matcher_concurrency
    if concurrency > _MAX_CONCURRENCY:
        print(
            f"WARNING: --concurrency {concurrency} exceeds the matcher's connection pool "
            f"(max_size=5, synchronous routes); clamping to {_MAX_CONCURRENCY}",
            file=sys.stderr,
            flush=True,
        )
        concurrency = _MAX_CONCURRENCY
    concurrency = max(1, concurrency)
    if concurrency > 1:
        print(
            f"WARNING: --concurrency {concurrency} is above 1. The matcher embeds every "
            "query with a shared HuggingFace fast tokenizer that is not thread-safe, and "
            "returns 500 (RuntimeError: Already borrowed) when two queries race. Expect "
            "the run to abort.",
            file=sys.stderr,
            flush=True,
        )

    client = build_matcher_client(settings, base_url=args.matcher_base_url)
    try:
        # Preflight before the first query: a run that cannot say which index
        # answered is not reproducible, and a matcher with no index should cost
        # one second rather than one timeout per row.
        try:
            index = client.index_status()
        except MatcherError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            f"matcher {client.base_url}: {index.get('embedding_model')} "
            f"dim {index.get('embedding_dim')}, {index.get('document_count')} documents "
            f"from {index.get('sources')}",
            file=sys.stderr,
            flush=True,
        )
        manifest = read_manifest(Path(args.benchmark))
        _warn_on_index_drift(manifest, index, args.benchmark)
        report = evaluate(
            rows, top_k=args.top_k, find=client.match, concurrency=concurrency
        )
    finally:
        client.close()

    print(
        json.dumps(
            {"index": index, "label": args.label, "summary": report.summary},
            indent=2,
        )
    )
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
        "of sources. Both sources are sampled: the defaults give 25 supervisors + 25 postings "
        "= 150 rows. --supervisor-sample 0 / --posting-sample 0 take everything (the full "
        "posting corpus alone is 579 postings = 1737 rows, hours of work under --query-style "
        "llm). --seed drives both samples.",
    )
    gen.add_argument("--out", default="data/benchmark.jsonl")
    gen.add_argument("--sources", default="publications,postings")
    gen.add_argument(
        "--supervisor-names", nargs="*", help="explicit supervisor names; skips sampling"
    )
    gen.add_argument(
        "--supervisor-sample", type=int, default=25, help="supervisors to sample; 0 = all"
    )
    gen.add_argument(
        "--supervisor-min-pubs",
        type=int,
        default=3,
        help="publications a supervisor needs to qualify as a candidate",
    )
    gen.add_argument(
        "--supervisor-max-pubs",
        type=int,
        default=20,
        help="most recent publications fed to KeyBERT per supervisor; 0 = all",
    )
    gen.add_argument(
        "--require-cris-uuid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only supervisors whose authority is CRIS-typed, i.e. that resolve "
        "to a person row. Drops nothing on the current corpus -- backend-core "
        "already restricts uzh_authors to registered UZH researchers -- so the "
        "printed candidate/cris-typed pair is really an invariant check",
    )
    gen.add_argument(
        "--posting-sample", type=int, default=25, help="postings to sample; 0 = all"
    )
    gen.add_argument(
        "--publication-language",
        default=None,
        help="restrict publications to one ISO 639-3 language, e.g. eng or deu; "
        "default is all languages. Does not affect postings, which carry no language label",
    )
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument(
        "--keyword-top-n",
        type=int,
        default=None,
        help="keyphrases per source; overrides KEYBERT_TOP_N",
    )
    gen.add_argument("--query-style", choices=("auto", "template", "llm"), default="auto")
    gen.add_argument(
        "--difficulty-threshold",
        type=int,
        default=10,
        help="publication count at or above which a supervisor is tagged 'easy'; "
        "below is 'hard'. Postings are always 'easy'. Default 10 (between "
        "--supervisor-min-pubs 3 and --supervisor-max-pubs 20)",
    )
    gen.add_argument(
        "--specificity",
        action="store_true",
        default=False,
        help="specificity ablation mode: generates 3 rows per supervisor "
        "(general/adjacent/true) instead of the normal 3-complexity output. "
        "Requires --supervisor-names and --adjacent",
    )
    gen.add_argument(
        "--adjacent",
        action="append",
        default=None,
        help="'SupervisorName=adjacent subfield' for the specificity ablation. "
        "Repeatable, one per supervisor. Example: "
        "\"Sennrich, R=text summarisation\"",
    )
    gen.add_argument(
        "--ablation-background",
        action="store_true",
        default=False,
        help="background ablation mode: generates 4 arms per supervisor "
        "(topic_only / plus_program / plus_background / full) to test whether "
        "student context helps or hurts retrieval",
    )
    _add_log_level(gen)
    _add_renamed_flags(gen)

    ev = sub.add_parser("evaluate", help="trial retrieval: hit@k / MRR for the gold supervisors")
    ev.add_argument("--benchmark", default="data/benchmark.jsonl")
    ev.add_argument("--top-k", type=int, default=10)
    ev.add_argument("--out", default=None, help="optional per-row results JSONL")
    ev.add_argument(
        "--matcher-base-url",
        default=None,
        help="matcher HTTP API to evaluate against; overrides MATCHER_BASE_URL, so a "
        "run can be pointed at a differently configured matcher without editing .env",
    )
    ev.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=f"concurrent /v1/match calls (default MATCHER_CONCURRENCY=1). Clamped to "
        f"{_MAX_CONCURRENCY} by the matcher's connection pool, but anything above 1 is "
        "currently unsafe: the matcher embeds queries with a shared, non-thread-safe "
        "tokenizer and returns 500 (RuntimeError: Already borrowed) under concurrency",
    )
    ev.add_argument(
        "--label",
        default=None,
        help="free-text note recorded in the report, e.g. the matcher's retrieval "
        "regime. GET /v1/index/status reports the index but not whether the server "
        "was configured to require a UZH author, so that has to be recorded here",
    )
    _add_log_level(ev)

    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    if args.command == "generate":
        return _generate(args)
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
