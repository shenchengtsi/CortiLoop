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
    python -m benchmarks.longmemeval                    # offline (local mode)
    python -m benchmarks.longmemeval --provider openai  # with OpenAI
    python -m benchmarks.longmemeval --provider ollama  # with Ollama
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
    retain_ms: float = 0.0
    recall_ms: float = 0.0
    details: str = ""
    retrieved: list[str] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    """Complete benchmark results."""
    results: list[BenchmarkResult] = field(default_factory=list)
    provider: str = "local"

    @property
    def summary(self) -> dict[str, Any]:
        categories: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r)

        summary = {}
        for cat, results in categories.items():
            passed = sum(1 for r in results if r.passed)
            avg_score = sum(r.score for r in results) / len(results) if results else 0
            avg_retain = sum(r.retain_ms for r in results) / len(results) if results else 0
            avg_recall = sum(r.recall_ms for r in results) / len(results) if results else 0
            summary[cat] = {
                "passed": passed,
                "total": len(results),
                "accuracy": round(passed / len(results), 3) if results else 0,
                "avg_score": round(avg_score, 3),
                "avg_retain_ms": round(avg_retain, 1),
                "avg_recall_ms": round(avg_recall, 1),
            }

        total_passed = sum(1 for r in self.results if r.passed)
        total_retain = sum(r.retain_ms for r in self.results)
        total_recall = sum(r.recall_ms for r in self.results)
        n = len(self.results) or 1
        summary["overall"] = {
            "passed": total_passed,
            "total": len(self.results),
            "accuracy": round(total_passed / n, 3),
            "avg_retain_ms": round(total_retain / n, 1),
            "avg_recall_ms": round(total_recall / n, 1),
        }
        return summary

    def print_report(self):
        CAT_NAMES = {
            "IE": "Information Extraction",
            "TR": "Temporal Reasoning",
            "KU": "Knowledge Update",
            "AR": "Associative Retrieval",
            "MSR": "Multi-Session Reasoning",
        }

        print()
        print("=" * 72)
        print(f"  CortiLoop LongMemEval Benchmark   (provider: {self.provider})")
        print("=" * 72)

        s = self.summary
        for cat in ["IE", "TR", "KU", "AR", "MSR"]:
            if cat not in s:
                continue
            info = s[cat]
            pct = info["accuracy"]
            bar = "#" * int(pct * 20) + "-" * (20 - int(pct * 20))
            name = CAT_NAMES.get(cat, cat)
            print(f"\n  {cat}  [{bar}]  {info['passed']}/{info['total']}  "
                  f"({pct:.0%})  score={info['avg_score']:.3f}")
            print(f"       {name}")
            print(f"       retain={info['avg_retain_ms']:.1f}ms  recall={info['avg_recall_ms']:.1f}ms")

        overall = s.get("overall", {})
        print(f"\n{'─' * 72}")
        acc = overall.get("accuracy", 0)
        print(f"  Overall: {overall.get('passed', 0)}/{overall.get('total', 0)} ({acc:.0%})")
        print(f"  Avg latency:  retain={overall.get('avg_retain_ms', 0):.1f}ms  "
              f"recall={overall.get('avg_recall_ms', 0):.1f}ms")
        print("=" * 72)

        # Per-case details
        print("\n  Per-Case Results:")
        print(f"  {'ID':<8} {'Status':<6} {'Score':>6}  {'Description'}")
        print(f"  {'─'*8} {'─'*6} {'─'*6}  {'─'*40}")
        for r in self.results:
            status = " PASS" if r.passed else " FAIL"
            print(f"  {r.case_id:<8} {status:<6} {r.score:>5.2f}  {r.description}")
            if not r.passed and r.details:
                # Wrap details
                detail_lines = r.details[:120]
                print(f"           -> {detail_lines}")

        print()


# ── Benchmark Cases ──

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
        "desc": "Filter noise (should ignore greetings)",
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
        "retain": ["张三是产品经理，负责用户增长项目，使用飞书和Notion做项目管理"],
        "query": "张三负责什么项目",
        "expect_contains": ["用户增长"],
    },
]

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
        "query": "What version was released in March?",
        "expect_contains": ["v2.0"],
    },
]

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
            "不对，API timeout应该是60 seconds",
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

MSR_CASES = [
    {
        "id": "MSR-01",
        "desc": "Cross-session synthesis",
        "retain": [
            "Session 1: Alice said the deadline is end of Q2",
            "Session 2: Bob said Q2 deliverables include auth refactor and API v3",
            "Session 3: The auth refactor is 80% complete",
        ],
        "query": "What is the status of the auth refactor?",
        "expect_contains": ["80%"],
    },
    {
        "id": "MSR-02",
        "desc": "Pattern detection across sessions",
        "retain": [
            "User asked to format code with prettier",
            "User asked to format code with prettier again",
            "User asked to auto-format on save",
        ],
        "query": "What tool does the user use for formatting?",
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


def _create_engine(config: CortiLoopConfig, provider: str) -> CortiLoop:
    """Create engine, injecting LocalLLMClient when provider is 'local'."""
    if provider == "local":
        from cortiloop.llm.local_client import LocalLLMClient
        local = LocalLLMClient(embedding_dim=config.llm.embedding_dim)
        return CortiLoop(config, llm=local)

    return CortiLoop(config)


async def _run_case(
    config: CortiLoopConfig,
    provider: str,
    category: str,
    case: dict,
) -> BenchmarkResult:
    """Run a single benchmark case."""
    loop = _create_engine(config, provider)

    try:
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
        passed = score >= 0.5

        details = ""
        retrieved = [r["content"][:80] for r in results[:5]]
        if not passed:
            missing = [e for e in expected if e.lower() not in all_content]
            details = f"Missing: {missing}. Got: {retrieved}"

        return BenchmarkResult(
            category=category,
            case_id=case["id"],
            description=case["desc"],
            passed=passed,
            score=score,
            latency_ms=retain_ms + recall_ms,
            retain_ms=retain_ms,
            recall_ms=recall_ms,
            details=details,
            retrieved=retrieved,
        )
    finally:
        loop.close()


async def run_benchmark(
    config: CortiLoopConfig | None = None,
    provider: str = "local",
    categories: list[str] | None = None,
) -> BenchmarkSuite:
    """
    Run the full LongMemEval benchmark suite.

    Args:
        config: CortiLoop configuration (default: in-memory SQLite)
        provider: LLM provider ("local" for offline, "openai", "ollama", etc.)
        categories: Which categories to run (default: all)

    Returns:
        BenchmarkSuite with results
    """
    config = config or CortiLoopConfig(db_path=":memory:")
    cats = categories or list(ALL_CASES.keys())
    suite = BenchmarkSuite(provider=provider)

    for cat in cats:
        cases = ALL_CASES.get(cat, [])
        for case in cases:
            try:
                result = await _run_case(config, provider, cat, case)
                suite.results.append(result)
                status = "PASS" if result.passed else "FAIL"
                logger.info(
                    "  %s  [%s] %s: score=%.2f  retain=%.0fms recall=%.0fms",
                    status, cat, case["id"], result.score,
                    result.retain_ms, result.recall_ms,
                )
            except Exception as e:
                logger.error("  ERROR [%s] %s: %s", cat, case["id"], e)
                suite.results.append(BenchmarkResult(
                    category=cat,
                    case_id=case["id"],
                    description=case["desc"],
                    passed=False,
                    score=0.0,
                    latency_ms=0,
                    details=f"Error: {e}",
                ))

    return suite


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="CortiLoop LongMemEval Benchmark")
    parser.add_argument("--db", default=":memory:", help="Database path (default: in-memory)")
    parser.add_argument("--categories", nargs="*", help="Categories to run (IE TR KU AR MSR)")
    parser.add_argument("--provider", default="local",
                        help="LLM provider: local (offline), openai, ollama, anthropic, litellm")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model (ignored for local)")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dimension (default: 256 for local, 1536 for openai)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = CortiLoopConfig(db_path=args.db)
    config.llm.provider = args.provider
    config.llm.model = args.model

    if args.provider == "local":
        config.llm.embedding_dim = args.embedding_dim
    elif args.provider == "ollama":
        config.llm.embedding_dim = 768
        config.llm.embedding_model = "nomic-embed-text"

    suite = await run_benchmark(config, provider=args.provider, categories=args.categories)

    if args.json:
        output = {
            "provider": args.provider,
            "model": args.model if args.provider != "local" else "local-rules",
            "summary": suite.summary,
            "cases": [
                {
                    "id": r.case_id, "category": r.category,
                    "passed": r.passed, "score": r.score,
                    "retain_ms": round(r.retain_ms, 1),
                    "recall_ms": round(r.recall_ms, 1),
                    "retrieved": r.retrieved,
                }
                for r in suite.results
            ],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        suite.print_report()


if __name__ == "__main__":
    asyncio.run(main())
