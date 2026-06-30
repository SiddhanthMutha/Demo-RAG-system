"""CLI entry point for running evaluations via `python -m src.evaluation.cli`."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.evaluation.harness import run_evaluation


def _load_dataset(path: str):
    """Load a QA dataset from a JSON file."""
    p = Path(path)
    if not p.exists():
        print(f"Dataset file not found: {p}", file=sys.stderr)
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _build_synthetic_dataset():
    """Generate a synthetic dataset from ingested documents."""
    from src.evaluation.dataset import generate_synthetic_qa
    from src.database import AsyncSessionLocal
    from src.database.models import ChunkRecord
    from sqlalchemy import select

    async def _inner():
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(ChunkRecord).limit(50))
            chunks = result.scalars().all()
            from src.models import Chunk, DocumentType
            pydantic_chunks = []
            for c in chunks:
                pydantic_chunks.append(
                    Chunk(
                        id=c.id,
                        document_id=c.document_id,
                        content=c.content,
                        token_count=c.token_count,
                        chunk_index=c.chunk_index,
                        metadata=c.chunk_metadata or {},
                    )
                )
            qa_pairs = await generate_synthetic_qa(pydantic_chunks, max_qa_pairs=20)
            return qa_pairs

    return asyncio.run(_inner())


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI")
    parser.add_argument(
        "command",
        choices=["run"],
        help="Command to execute: 'run' an evaluation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to JSON dataset file",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Auto-generate QA dataset from ingested documents",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K for retrieval evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (JSON)",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        if args.synthetic:
            print("Generating synthetic QA dataset...")
            qa_pairs = _build_synthetic_dataset()
            queries = [
                {
                    "query": qa["question"],
                    "expected_ids": [qa["chunk_id"]],
                    "expected_answer": qa["answer"],
                }
                for qa in qa_pairs
            ]
        elif args.dataset:
            queries = _load_dataset(args.dataset)
        else:
            print("Error: provide --dataset <path> or --synthetic", file=sys.stderr)
            sys.exit(1)

        print(f"Running evaluation on {len(queries)} queries (top_k={args.top_k})...")
        results = asyncio.run(run_evaluation(queries, top_k=args.top_k))

        output = {
            "precision_at_k": results.get("precision_at_k"),
            "recall_at_k": results.get("recall_at_k"),
            "mrr": results.get("mrr"),
            "ndcg": results.get("ndcg"),
            "correctness_rate": results.get("correctness_rate"),
            "faithfulness_rate": results.get("faithfulness_rate"),
            "avg_latency_ms": results.get("avg_latency_ms"),
            "total_cost_usd": results.get("total_cost_usd"),
            "num_queries": len(results.get("queries", [])),
        }

        if args.output:
            Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"Results written to {args.output}")
        else:
            print(json.dumps(output, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
