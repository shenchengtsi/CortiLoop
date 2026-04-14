"""
LongMemEval Official Benchmark — 500 questions from the academic dataset.

Loads the official LongMemEval dataset (ICLR 2025, arxiv 2410.10813) and
evaluates CortiLoop across 6 question types:

  1. single-session-user       — Extract facts from user messages
  2. single-session-assistant   — Extract facts from assistant messages
  3. single-session-preference  — Infer user preferences from conversation
  4. temporal-reasoning         — Time-based reasoning across sessions
  5. knowledge-update           — Handle corrected/updated information
  6. multi-session              — Synthesize across multiple sessions

Supports 3 dataset variants:
  - oracle:  Answer-relevant sessions only (fast, for debugging)
  - s:       ~40 sessions per question, ~115K tokens (recommended)
  - m:       ~500 sessions per question, ~1.5M tokens (stress test)

Usage:
    # First download:
    python -m benchmarks.download_longmemeval --variant s

    # Run with LLM (recommended for answer generation + judging):
    python -m benchmarks.longmemeval_official --variant s --provider openai

    # Run offline (string-matching only, no LLM judge):
    python -m benchmarks.longmemeval_official --variant oracle --provider local

    # Run a subset:
    python -m benchmarks.longmemeval_official --variant s --max-items 20

    # Filter by question type:
    python -m benchmarks.longmemeval_official --variant s --types knowledge-update temporal-reasoning
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cortiloop import CortiLoop, CortiLoopConfig

logger = logging.getLogger("cortiloop.benchmark.official")

DATA_DIR = Path(__file__).parent / "data"

VARIANT_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

# Human-readable names for question types
QTYPE_NAMES = {
    "single-session-user": "Single-Session (User)",
    "single-session-assistant": "Single-Session (Assistant)",
    "single-session-preference": "Single-Session (Preference)",
    "temporal-reasoning": "Temporal Reasoning",
    "knowledge-update": "Knowledge Update",
    "multi-session": "Multi-Session",
}


# ── Data structures ──


@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    generated_answer: str
    recall_results: list[dict[str, Any]]
    score: float          # 0.0-1.0
    passed: bool
    judge_reasoning: str  # LLM judge explanation or match details
    ingest_ms: float      # time to retain all sessions
    recall_ms: float      # time to recall
    answer_ms: float      # time to generate answer
    judge_ms: float       # time to judge answer
    num_sessions: int     # how many sessions were ingested


@dataclass
class BenchmarkReport:
    """Full benchmark results."""
    variant: str
    provider: str
    model: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, Any]:
        by_type: dict[str, list[QuestionResult]] = {}
        for r in self.results:
            by_type.setdefault(r.question_type, []).append(r)

        summary = {}
        for qtype, results in sorted(by_type.items()):
            passed = sum(1 for r in results if r.passed)
            n = len(results)
            avg_score = sum(r.score for r in results) / n if n else 0
            avg_ingest = sum(r.ingest_ms for r in results) / n if n else 0
            avg_recall = sum(r.recall_ms for r in results) / n if n else 0
            summary[qtype] = {
                "passed": passed,
                "total": n,
                "accuracy": round(passed / n, 3) if n else 0,
                "avg_score": round(avg_score, 3),
                "avg_ingest_ms": round(avg_ingest, 1),
                "avg_recall_ms": round(avg_recall, 1),
            }

        total_passed = sum(1 for r in self.results if r.passed)
        n = len(self.results) or 1
        summary["overall"] = {
            "passed": total_passed,
            "total": len(self.results),
            "accuracy": round(total_passed / n, 3),
            "avg_ingest_ms": round(sum(r.ingest_ms for r in self.results) / n, 1),
            "avg_recall_ms": round(sum(r.recall_ms for r in self.results) / n, 1),
        }
        return summary

    def print_report(self):
        print()
        print("=" * 78)
        print(f"  CortiLoop × LongMemEval Official Benchmark")
        print(f"  variant={self.variant}  provider={self.provider}  model={self.model}")
        print("=" * 78)

        s = self.summary
        for qtype in QTYPE_NAMES:
            if qtype not in s:
                continue
            info = s[qtype]
            pct = info["accuracy"]
            bar = "#" * int(pct * 20) + "-" * (20 - int(pct * 20))
            name = QTYPE_NAMES.get(qtype, qtype)
            print(f"\n  [{bar}]  {info['passed']}/{info['total']}  ({pct:.0%})")
            print(f"    {name}")
            print(f"    ingest={info['avg_ingest_ms']:.0f}ms  recall={info['avg_recall_ms']:.0f}ms")

        overall = s.get("overall", {})
        print(f"\n{'─' * 78}")
        acc = overall.get("accuracy", 0)
        print(f"  Overall: {overall.get('passed', 0)}/{overall.get('total', 0)} ({acc:.0%})")
        print(f"  Avg latency:  ingest={overall.get('avg_ingest_ms', 0):.0f}ms  "
              f"recall={overall.get('avg_recall_ms', 0):.0f}ms")
        print("=" * 78)

        # Failed questions detail
        failed = [r for r in self.results if not r.passed]
        if failed:
            print(f"\n  Failed Questions ({len(failed)}):")
            print(f"  {'ID':<12} {'Type':<28} {'Score':>5}  Question (truncated)")
            print(f"  {'─'*12} {'─'*28} {'─'*5}  {'─'*30}")
            for r in failed[:30]:  # show first 30 failures
                q_short = r.question[:50] + "..." if len(r.question) > 50 else r.question
                print(f"  {r.question_id:<12} {r.question_type:<28} {r.score:>5.2f}  {q_short}")

        print()


# ── Dataset loading ──


def load_dataset(variant: str, max_items: int | None = None,
                 types: list[str] | None = None) -> list[dict]:
    """Load LongMemEval dataset JSON."""
    filename = VARIANT_FILES.get(variant)
    if not filename:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANT_FILES.keys())}")

    path = DATA_DIR / filename
    if not path.exists():
        print(f"Dataset not found: {path}", file=sys.stderr)
        print(f"Run: python -m benchmarks.download_longmemeval --variant {variant}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        dataset = json.load(f)

    # Filter by question type
    if types:
        dataset = [item for item in dataset if item.get("question_type") in types]

    if max_items:
        dataset = dataset[:max_items]

    return dataset


def parse_date(date_str: str) -> datetime:
    """Parse LongMemEval date string (e.g., '2023/05/20 (Sat) 02:21')."""
    # Strip day-of-week parenthetical
    cleaned = date_str.split("(")[0].strip() if "(" in date_str else date_str

    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        logger.warning("Failed to parse date: %s, using now()", date_str)
        return datetime.now(timezone.utc)


def format_session_as_text(session_turns: list[dict]) -> str:
    """Convert a list of conversation turns into readable text for retain()."""
    lines = []
    for turn in session_turns:
        if isinstance(turn, dict):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        elif isinstance(turn, str):
            lines.append(turn)
    return "\n".join(lines)


# ── Answer generation ──


async def generate_answer_llm(
    question: str,
    recall_results: list[dict[str, Any]],
    llm,
    question_date: str | None = None,
) -> str:
    """Generate answer from recalled memories using the LLM."""
    # Build context from recall results — sort by date (newest first) for temporal clarity
    dated = [(r, r.get("session_timestamp", "")) for r in recall_results]
    dated.sort(key=lambda x: x[1] if x[1] else "", reverse=True)

    context_parts = []
    for i, (r, _) in enumerate(dated, 1):
        content = r.get("content", "")
        score = r.get("score", 0)
        mem_type = r.get("type", "unknown")
        ts = r.get("session_timestamp", "")
        ts_label = f", date={ts}" if ts else ""
        context_parts.append(f"[Memory {i}] (type={mem_type}, relevance={score:.3f}{ts_label})\n{content}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant memories found."

    date_info = f"\nCurrent date: {question_date}" if question_date else ""

    system_prompt = (
        "You are a helpful assistant answering questions based on retrieved memories "
        "from previous conversations. Answer concisely based only on the provided context.\n\n"
        "Important rules:\n"
        "- When memories contain CONFLICTING information about the same topic (e.g. different numbers, "
        "different statuses, different locations), ALWAYS use the MORE RECENT one. "
        "Memories are sorted newest-first; trust the first (most recent) version you see.\n"
        "- For counting or totaling questions: list EVERY individual item found separately, "
        "number them, then give the total. Do NOT merge items unless they are EXACTLY the same. "
        "Different events, purchases, or items mentioned in separate memories are DISTINCT.\n"
        "- If the context contains ANY relevant information, use it to give the BEST answer possible. "
        "Only say 'not enough information' if the context contains ZERO relevant information about the question.\n"
        "- When the question asks for a recommendation or suggestion, use any information about user "
        "preferences, habits, or past experiences found in the memories to personalize your answer."
    )

    user_prompt = f"""Based on the following retrieved memories, answer the question.
{date_info}

Retrieved memories:
{context}

Question: {question}

Answer concisely and directly."""

    try:
        response = await llm.complete(system_prompt, user_prompt, response_format="text")
        # Handle case where response is JSON
        if response.strip().startswith("{"):
            try:
                parsed = json.loads(response)
                return parsed.get("answer", parsed.get("result", response))
            except json.JSONDecodeError:
                pass
        return response.strip()
    except Exception as e:
        logger.warning("Answer generation failed: %s", e)
        # Fallback: concatenate top recall results
        return " ".join(r.get("content", "") for r in recall_results[:3])


def generate_answer_local(
    question: str,
    recall_results: list[dict[str, Any]],
) -> str:
    """Generate answer from recalled memories without LLM (concatenate top results)."""
    return " | ".join(r.get("content", "")[:200] for r in recall_results[:5])


# ── Evaluation / Judging ──


async def judge_answer_llm(
    question: str,
    gold_answer: str,
    generated_answer: str,
    llm,
) -> tuple[float, str]:
    """Use LLM to judge if generated answer matches gold answer. Returns (score, reasoning)."""
    system_prompt = (
        "You are an evaluation judge. Compare the generated answer to the gold answer "
        "and determine if they are semantically equivalent. Score from 0.0 to 1.0:\n"
        "  1.0 = Fully correct, captures all key information\n"
        "  0.75 = Mostly correct, minor details missing\n"
        "  0.5 = Partially correct, some key info present\n"
        "  0.25 = Marginally relevant, mostly wrong\n"
        "  0.0 = Completely wrong or irrelevant\n\n"
        "Respond in JSON: {\"score\": <float>, \"reasoning\": \"<explanation>\"}"
    )

    user_prompt = f"""Question: {question}

Gold answer: {gold_answer}

Generated answer: {generated_answer}

Judge the generated answer against the gold answer. Respond in JSON."""

    try:
        response = await llm.complete(system_prompt, user_prompt, response_format="json")
        parsed = json.loads(response)
        score = float(parsed.get("score", 0))
        reasoning = parsed.get("reasoning", "")
        return min(max(score, 0.0), 1.0), reasoning
    except Exception as e:
        logger.warning("LLM judge failed: %s, falling back to string match", e)
        return judge_answer_string(gold_answer, generated_answer)


def judge_answer_string(gold_answer: str, generated_answer: str) -> tuple[float, str]:
    """Simple string-matching judge as fallback."""
    gold_lower = gold_answer.lower().strip()
    gen_lower = generated_answer.lower().strip()

    # Handle list answers (gold may be a JSON list)
    gold_items = []
    if gold_lower.startswith("["):
        try:
            gold_items = [str(x).lower().strip() for x in json.loads(gold_answer)]
        except (json.JSONDecodeError, TypeError):
            gold_items = [gold_lower]
    else:
        gold_items = [gold_lower]

    # Check what fraction of gold items appear in generated answer
    found = 0
    for item in gold_items:
        # Also check with flexible matching (remove quotes, articles)
        clean_item = re.sub(r'^(the|a|an)\s+', '', item).strip().strip('"\'')
        if clean_item in gen_lower or item in gen_lower:
            found += 1

    if not gold_items:
        return (0.0, "Empty gold answer")

    score = found / len(gold_items)
    detail = f"Matched {found}/{len(gold_items)} gold items"
    return (score, detail)


# ── Main benchmark runner ──


async def run_single_question(
    item: dict,
    config: CortiLoopConfig,
    provider: str,
    use_llm_judge: bool = True,
) -> QuestionResult:
    """Run benchmark on a single LongMemEval question item."""
    question_id = item.get("question_id", "unknown")
    question_type = item.get("question_type", "unknown")
    question = item.get("question", "")
    gold_answer = item.get("answer", "")
    question_date = item.get("question_date")

    # Normalize gold_answer to string
    if isinstance(gold_answer, list):
        gold_answer = json.dumps(gold_answer)
    else:
        gold_answer = str(gold_answer)

    sessions = item.get("haystack_sessions", [])
    dates = item.get("haystack_dates", [])
    session_ids = item.get("haystack_session_ids", [])

    # Align list lengths
    min_len = min(len(sessions), len(dates), len(session_ids))
    sessions = sessions[:min_len]
    dates = dates[:min_len]
    session_ids = session_ids[:min_len]

    # Create fresh CortiLoop instance for this question (isolation)
    loop = _create_engine(config, provider)

    try:
        # ── Phase 1: Ingest sessions ──
        t0 = time.perf_counter()
        for session_turns, date_str, sid in zip(sessions, dates, session_ids):
            # Clean turns — remove has_answer metadata
            cleaned = []
            for turn in session_turns:
                if isinstance(turn, dict):
                    cleaned.append({k: v for k, v in turn.items() if k != "has_answer"})
                else:
                    cleaned.append(turn)

            text = format_session_as_text(cleaned)
            session_date = parse_date(date_str) if date_str else None

            # Retain the session text with context
            context = f"Conversation session {sid}"
            if session_date:
                context += f" on {session_date.strftime('%Y-%m-%d')}"

            await loop.retain(
                text,
                session_id=str(sid),
                task_context=context,
                session_timestamp=session_date,
            )

        ingest_ms = (time.perf_counter() - t0) * 1000

        # ── Phase 2: Recall ──
        # Aggregation questions need more results to avoid missing items
        _AGG_PATTERN = re.compile(
            r"how many|how much|total|in total|combined|altogether",
            re.IGNORECASE,
        )
        recall_k = 100 if _AGG_PATTERN.search(question) else 20

        t0 = time.perf_counter()
        recall_results = await loop.recall(question, top_k=recall_k)
        recall_ms = (time.perf_counter() - t0) * 1000

        # ── Phase 3: Generate answer ──
        t0 = time.perf_counter()
        if provider == "local":
            generated_answer = generate_answer_local(question, recall_results)
        else:
            generated_answer = await generate_answer_llm(
                question, recall_results, loop.llm, question_date,
            )
        answer_ms = (time.perf_counter() - t0) * 1000

        # ── Phase 4: Judge answer ──
        t0 = time.perf_counter()
        if use_llm_judge and provider != "local":
            score, reasoning = await judge_answer_llm(
                question, gold_answer, generated_answer, loop.llm,
            )
        else:
            score, reasoning = judge_answer_string(gold_answer, generated_answer)
        judge_ms = (time.perf_counter() - t0) * 1000

        return QuestionResult(
            question_id=question_id,
            question_type=question_type,
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
            recall_results=recall_results[:5],  # keep top 5 for reporting
            score=score,
            passed=score >= 0.5,
            judge_reasoning=reasoning,
            ingest_ms=ingest_ms,
            recall_ms=recall_ms,
            answer_ms=answer_ms,
            judge_ms=judge_ms,
            num_sessions=len(sessions),
        )

    except Exception as e:
        logger.error("Error on question %s: %s", question_id, e)
        return QuestionResult(
            question_id=question_id,
            question_type=question_type,
            question=question,
            gold_answer=gold_answer,
            generated_answer="",
            recall_results=[],
            score=0.0,
            passed=False,
            judge_reasoning=f"Error: {e}",
            ingest_ms=0,
            recall_ms=0,
            answer_ms=0,
            judge_ms=0,
            num_sessions=len(sessions),
        )
    finally:
        loop.close()


def _create_engine(config: CortiLoopConfig, provider: str) -> CortiLoop:
    """Create CortiLoop engine, using LocalLLMClient for 'local' provider."""
    if provider == "local":
        from cortiloop.llm.local_client import LocalLLMClient
        local = LocalLLMClient(embedding_dim=config.llm.embedding_dim)
        return CortiLoop(config, llm=local)
    return CortiLoop(config)


async def run_benchmark(
    variant: str = "s",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    max_items: int | None = None,
    types: list[str] | None = None,
    use_llm_judge: bool = True,
    concurrency: int = 1,
    db_path: str = ":memory:",
    embedding_dim: int = 256,
    base_url: str = "",
    api_key: str = "",
) -> BenchmarkReport:
    """
    Run the full LongMemEval official benchmark.

    Args:
        variant: Dataset variant ("oracle", "s", "m")
        provider: LLM provider ("local", "openai", "ollama", "anthropic", "litellm")
        model: LLM model name
        max_items: Max questions to evaluate (None = all 500)
        types: Filter to specific question types
        use_llm_judge: Use LLM for answer evaluation (else string-match)
        concurrency: Parallel question evaluation (1 = sequential)
        db_path: Database path for CortiLoop instances
        embedding_dim: Embedding dimension (for local provider)

    Returns:
        BenchmarkReport with all results
    """
    dataset = load_dataset(variant, max_items, types)
    report = BenchmarkReport(variant=variant, provider=provider, model=model)

    total = len(dataset)
    logger.info("Loaded %d questions (variant=%s)", total, variant)

    # Type distribution
    type_counts: dict[str, int] = {}
    for item in dataset:
        qt = item.get("question_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1
    for qt, count in sorted(type_counts.items()):
        logger.info("  %s: %d questions", qt, count)

    sem = asyncio.Semaphore(concurrency)

    async def run_with_semaphore(idx: int, item: dict) -> QuestionResult:
        async with sem:
            qid = item.get("question_id", "?")
            logger.info("[%d/%d] Running %s (%s)...", idx + 1, total, qid,
                        item.get("question_type", "?"))

            config = CortiLoopConfig(db_path=db_path)
            config.llm.provider = provider
            config.llm.model = model
            if base_url:
                config.llm.base_url = base_url
            if api_key:
                config.llm.api_key = api_key
            config.llm.embedding_dim = embedding_dim
            if provider == "ollama":
                config.llm.embedding_model = "nomic-embed-text"

            # Disable attention gate for benchmark (ingest everything)
            config.attention_gate.enabled = False

            # Enable cross-encoder reranking for better retrieval precision
            config.retrieval.rerank_enabled = True
            config.retrieval.rerank_top_k = 80

            result = await run_single_question(item, config, provider, use_llm_judge)

            status = "PASS" if result.passed else "FAIL"
            logger.info("  %s  score=%.2f  ingest=%0.fms  recall=%.0fms",
                        status, result.score, result.ingest_ms, result.recall_ms)
            return result

    if concurrency == 1:
        # Sequential — simpler logging
        for idx, item in enumerate(dataset):
            result = await run_with_semaphore(idx, item)
            report.results.append(result)
    else:
        # Concurrent
        tasks = [run_with_semaphore(idx, item) for idx, item in enumerate(dataset)]
        results = await asyncio.gather(*tasks)
        report.results.extend(results)

    return report


# ── CLI ──


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CortiLoop × LongMemEval Official Benchmark (500 questions)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download dataset first:
  python -m benchmarks.download_longmemeval --variant s

  # Run with OpenAI:
  python -m benchmarks.longmemeval_official --variant s --provider openai

  # Run offline (string-match only):
  python -m benchmarks.longmemeval_official --variant oracle --provider local

  # Run subset of question types:
  python -m benchmarks.longmemeval_official --variant s --types knowledge-update temporal-reasoning

  # Run first 20 questions only:
  python -m benchmarks.longmemeval_official --variant s --max-items 20
""",
    )
    parser.add_argument("--variant", choices=list(VARIANT_FILES.keys()), default="s",
                        help="Dataset variant (default: s)")
    parser.add_argument("--provider", default="openai",
                        help="LLM provider: local, openai, ollama, anthropic, litellm")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--base-url", default="",
                        help="Custom LLM API base URL (e.g. Volcengine, Ollama)")
    parser.add_argument("--api-key", default="",
                        help="LLM API key (overrides env var)")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Max questions to run (default: all)")
    parser.add_argument("--types", nargs="*", choices=list(QTYPE_NAMES.keys()),
                        help="Filter to specific question types")
    parser.add_argument("--db", default=":memory:",
                        help="Database path (default: in-memory)")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dim for local provider (default: 256)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallel question count (default: 1)")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Use string matching instead of LLM judge")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON results to file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    report = await run_benchmark(
        variant=args.variant,
        provider=args.provider,
        model=args.model,
        max_items=args.max_items,
        types=args.types,
        use_llm_judge=not args.no_llm_judge,
        concurrency=args.concurrency,
        db_path=args.db,
        embedding_dim=args.embedding_dim,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    # Output
    if args.json or args.output:
        output = {
            "variant": report.variant,
            "provider": report.provider,
            "model": report.model,
            "summary": report.summary,
            "questions": [
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "question": r.question,
                    "gold_answer": r.gold_answer,
                    "generated_answer": r.generated_answer,
                    "score": r.score,
                    "passed": r.passed,
                    "judge_reasoning": r.judge_reasoning,
                    "ingest_ms": round(r.ingest_ms, 1),
                    "recall_ms": round(r.recall_ms, 1),
                    "answer_ms": round(r.answer_ms, 1),
                    "num_sessions": r.num_sessions,
                }
                for r in report.results
            ],
        }

        json_str = json.dumps(output, indent=2, ensure_ascii=False)

        if args.output:
            Path(args.output).write_text(json_str)
            print(f"Results saved to {args.output}")

        if args.json:
            print(json_str)
    else:
        report.print_report()


if __name__ == "__main__":
    asyncio.run(main())
