"""
Synthetic QA dataset generation.
Uses the LLM to generate question-answer pairs from ingested document chunks.
"""
import asyncio
import random
from typing import Any, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from src.generation.llm_client import LLMClient
from src.models import Chunk


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _generate_single_qa(chunk_text: str, llm: LLMClient) -> Dict[str, str]:
    """Generate a single Q&A pair from a document chunk."""
    prompt = (
        "Given the following text, generate a question and its answer. "
        "Respond in exactly this format (no extra text):\n"
        "QUESTION: <question>\n"
        "ANSWER: <answer>\n\n"
        f"TEXT:\n{chunk_text}\n"
    )
    response, _, _ = await llm.generate(prompt=prompt, system_prompt="", max_tokens=200)

    # Parse
    question = ""
    answer = ""
    for line in response.splitlines():
        if line.startswith("QUESTION:"):
            question = line[len("QUESTION:") :].strip()
        elif line.startswith("ANSWER:"):
            answer = line[len("ANSWER:") :].strip()

    return {"question": question, "answer": answer}


async def generate_synthetic_qa(
    chunks: List[Chunk],
    max_qa_pairs: int = 20,
    sample_size: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic Q&A pairs from document chunks.

    Args:
        chunks: List of Chunk objects to generate Q&A from.
        max_qa_pairs: Maximum number of Q&A pairs to generate.
        sample_size: Number of chunks to sample before generation.

    Returns:
        List of dicts with keys: question, answer, chunk_id, document_id
    """
    if not chunks:
        return []

    llm = LLMClient()
    sample_target = sample_size or max_qa_pairs
    selected = random.sample(chunks, min(sample_target, len(chunks)))

    tasks = []
    for chunk in selected:
        tasks.append(_generate_single_qa(chunk.content, llm))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    qa_pairs: List[Dict[str, Any]] = []
    for chunk, result in zip(selected, results):
        if isinstance(result, Exception):
            continue
        if result.get("question") and result.get("answer"):
            qa_pairs.append(
                {
                    "question": result["question"],
                    "answer": result["answer"],
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                }
            )

    return qa_pairs
