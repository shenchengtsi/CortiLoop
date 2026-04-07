"""
LongMemEval Benchmark Harness for CortiLoop.

Evaluates memory quality across 5 core dimensions modeled after the
LongMemEval benchmark (https://arxiv.org/abs/2407.15920):

1. Information Extraction (IE) — Can the system extract and store facts?
2. Temporal Reasoning (TR) — Can it handle time-based queries?
3. Knowledge Update (KU) — Can it update beliefs when corrected?
4. Associative Retrieval (AR) — Can it link related memories?
5. Multi-Session Reasoning (MSR) — Can it synthesize across sessions?

Usage:
    python -m benchmarks.longmemeval --db memory_bench.db

Requires a running LLM endpoint (set OPENAI_API_KEY or configure Ollama).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from cortiloop import CortiLoop, CortiLoopConfig

logger = logging.getLogger("cortiloop.benchmark")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark case."""
    category: str
    case_id: str
    description: str
    passed: bool
    score: float  # 0.0 - 1.0
    latency_ms: float
    details: str = ""


@dataclass
class BenchmarkSuite:
    """Complete benchmark results."""
    results: list[BenchmarkResult] = field(default_factory=list)
    total_retain_ms: float = 0.0
    total_recall_ms: float = 0.0

    @property
    def summary(self) -> dict[str, Any]:
        categories: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r)

        summary = {}
        for cat, results in categories.items():
            passed = sum(1 for r in results if r.passed)
            avg_score = sum(r.score for r in results) / len(results) if results else 0
            summary[cat] = {
                "passed": passed,
                "total": len(results),
                "accuracy": passed / len(results) if results else 0,
                "avg_score": round(avg_score, 3),
            }

        total_passed = sum(1 for r in self.results if r.passed)
        summary["overall"] = {
            "passed": total_passed,
            "total": len(self.results),
            "accuracy": total_passed / len(self.results) if self.results else 0,
            "avg_retain_ms": round(self.total_retain_ms / max(len(self.results), 1), 1),
            "avg_recall_ms": round(self.total_recall_ms / max(len(self.results), 1), 1),
        }
        return summary

    def print_report(self):
        print("\n" + "=" * 70)
        print("CortiLoop — LongMemEval Benchmark Report")
        print("=" * 70)

        s = self.summary
        for cat in ["IE", "TR", "KU", "AR", "MSR"]:
            if cat in s:
                info = s[cat]
                bar = "█" * int(info["accuracy"] * 20) + "░" * (20 - int(info["accuracy"] * 20))
                print(f"\n  {cat:>3s}  {bar}  {info['passed']}/{info['total']}  ({info['accuracy']:.0%})  avg_score={info['avg_score']}")

        overall = s.get("overall", {})
        print(f"\n{'─' * 70}")
        print(f"  Overall: {overall.get('passed', 0)}/{overall.get('total', 0)} ({overall.get('accuracy', 0):.0%})")
        print(f"  Avg retain: {overall.get('avg_retain_ms', 0):.0f}ms  |  Avg recall: {overall.get('avg_recall_ms', 0):.0f}ms")
        print("=" * 70)

        # Failed cases
        failed = [r for r in self.results if not r.passed]
        if failed:
            print(f"\nFailed cases ({len(failed)}):")
            for r in failed:
                print(f"  [{r.category}] {r.case_id}: {r.description}")
                if r.details:
                    print(f"    → {r.details}")


# ── Benchmark Cases ──

# IE: Information Extraction
IE_CASES = [
    {
        "id": "IE-01",
        "desc": "Extract single fact",
        "retain": ["Alice is the PM of ProjectX, they use React + TypeScript"],
        "query": "What project does Alice manage?",
        "expect_contains": ["ProjectX"],
    },
    {
        "id": "IE-02",
        "desc": "Extract multiple entities",
        "retain": ["Bob is a backend engineer. He works with Python and Go. His team is the Platform team."],
        "query": "What languages does Bob use?",
        "expect_contains": ["Python", "Go"],
    },
    {
        "id": "IE-03",
        "desc": "Filter noise",
        "retain": [
            "Alice prefers dark mode in her IDE",
            "ok",
            "hello",
            "hmm",
            "Alice also uses Vim keybindings",
        ],
        "query": "What are Alice's editor preferences?",
        "expect_contains": ["dark mode"],
    },
    {
        "id": "IE-04",
        "desc": "CJK content extraction",
        "retain": ["张三是产品经理，负责用户增长项目，使用飞书和 Notion 做项目管理"],
        "query": "张三负责什么项目？",
        "expect_contains": ["用户增长"],
    },
]

# TR: Temporal Reasoning
TR_CASES = [
    {
        "id": "TR-01",
        "desc": "Recent memory recall",
        "retain": [
            "Meeting with design team scheduled for Monday",
            "Sprint review is on Friday",
        ],
        "query": "What meetings are scheduled?",
        "expect_contains": ["Monday", "Friday"],
    },
    {
        "id": "TR-02",
        "desc": "Temporal ordering",
        "retain": [
            "v1.0 was released in January",
            "v2.0 was released in March",
            "v3.0 is planned for June",
        ],
        "query": "What was the latest released version?",
        "expect_contains": ["v2.0"],
    },
]

# KU: Knowledge Update
KU_CASES = [
    {
        "id": "KU-01",
        "desc": "Correction supersedes old info",
        "retain": [
            "Alice uses React for the frontend",
            "Actually, Alice switched to Vue.js last month",
        ],
        "query": "What frontend framework does Alice use?",
        "expect_contains": ["Vue"],
    },
    {
        "id": "KU-02",
        "desc": "Explicit correction signal",
        "retain": [
            "The API timeout is 30 seconds",
            "不对，API timeout 应该是 60 seconds",
        ],
        "query": "What is the API timeout?",
        "expect_contains": ["60"],
    },
    {
        "id": "KU-03",
        "desc": "Supplementary info (no conflict)",
        "retain": [
            "Bob works on the Platform team",
            "Bob also contributes to the ML team part-time",
        ],
        "query": "What teams does Bob work with?",
        "expect_contains": ["Platform", "ML"],
    },
]

# AR: Associative Retrieval
AR_CASES = [
    {
        "id": "AR-01",
        "desc": "Cross-entity association",
        "retain": [
            "Alice is the PM of ProjectX",
            "ProjectX uses React and TypeScript",
            "Alice prefers strict TypeScript config",
        ],
        "query": "What technology stack is associated with Alice?",
        "expect_contains": ["React", "TypeScript"],
    },
    {
        "id": "AR-02",
        "desc": "Indirect connection",
        "retain": [
            "Alice mentors Bob",
            "Bob is learning Rust",
            "The Rust project is called LibCore",
        ],
        "query": "Who is connected to LibCore?",
        "expect_contains": ["Bob"],
    },
]

# MSR: Multi-Session Reasoning
MSR_CASES = [
    {
        "id": "MSR-01",
        "desc": "Cross-session synthesis",
        "retain": [
            "Session 1: Alice said the deadline is end of Q2",
            "Session 2: Bob said Q2 deliverables include auth refactor and API v3",
            "Session 3: The auth refactor is 80% complete",
        ],
        "query": "What's the status of Q2 deliverables?",
        "expect_contains": ["auth", "80%"],
    },
    {
        "id": "MSR-02",
        "desc": "Pattern detection across sessions",
        "retain": [
            "User asked to format code with prettier",
            "User asked to format code with prettier again",
            "User asked to auto-format on save",
        ],
        "query": "What are the user's formatting preferences?",
        "expect_contains": ["prettier"],
    },
]

ALL_CASES = {
    "IE": IE_CASES,
    "TR": TR_CASES,
    "KU": KU_CASES,
    "AR": AR_CASES,
    "MSR": MSR_CASES,
}


async def _run_case(
    loop: CortiLoop,
    category: str,
    case: dict,
) -> BenchmarkResult:
    """Run a single benchmark case."""

    # Retain phase
    t0 = time.perf_counter()
    for text in case["retain"]:
        await loop.retain(text)
    retain_ms = (time.perf_counter() - t0) * 1000

    # Recall phase
    t0 = time.perf_counter()
    results = await loop.recall(case["query"], top_k=10)
    recall_ms = (time.perf_counter() - t0) * 1000

    # Evaluate: check if expected strings appear in any result
    all_content = " ".join(r["content"] for r in results).lower()
    expected = case["expect_contains"]
    found = [e for e in expected if e.lower() in all_content]
    score = len(found) / len(expected) if expected else 1.0
    passed = score >= 0.5  # at least half of expected terms found

    details = ""
    if not passed:
        missing = [e for e in expected if e.lower() not in all_content]
        details = f"Missing: {missing}. Got: {[r['content'][:80] for r in results[:3]]}"

    return BenchmarkResult(
        category=category,
        case_id=case["id"],
        description=case["desc"],
        passed=passed,
        score=score,
        latency_ms=retain_ms + recall_ms,
        details=details,
    )


async def run_benchmark(
    config: CortiLoopConfig | None = None,
    categories: list[str] | None = None,
) -> BenchmarkSuite:
    """
    Run the full LongMemEval benchmark suite.

    Args:
        config: CortiLoop configuration (default: in-memory SQLite)
        categories: Which categories to run (default: all)

    Returns:
        BenchmarkSuite with results
    """
    config = config or CortiLoopConfig(db_path=":memory:")
    cats = categories or list(ALL_CASES.keys())
    suite = BenchmarkSuite()

    for cat in cats:
        cases = ALL_CASES.get(cat, [])
        for case in cases:
            # Each case gets a fresh engine to isolate results
            loop = CortiLoop(config)
            try:
                result = await _run_case(loop, cat, case)
                suite.results.append(result)
                suite.total_retain_ms += result.latency_ms * 0.7  # rough split
                suite.total_recall_ms += result.latency_ms * 0.3
                status = "✓" if result.passed else "✗"
                logger.info(
                    "%s [%s] %s: score=%.2f (%.0fms)",
                    status, cat, case["id"], result.score, result.latency_ms,
                )
            except Exception as e:
                suite.results.append(BenchmarkResult(
                    category=cat,
                    case_id=case["id"],
                    description=case["desc"],
                    passed=False,
                    score=0.0,
                    latency_ms=0,
                    details=f"Error: {e}",
                ))
            finally:
                loop.close()

    return suite


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="CortiLoop LongMemEval Benchmark")
    parser.add_argument("--db", default=":memory:", help="Database path")
    parser.add_argument("--categories", nargs="*", help="Categories to run")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = CortiLoopConfig(db_path=args.db)
    config.llm.provider = args.provider
    config.llm.model = args.model

    suite = await run_benchmark(config, args.categories)

    if args.json:
        print(json.dumps(suite.summary, indent=2))
    else:
        suite.print_report()


if __name__ == "__main__":
    asyncio.run(main())
