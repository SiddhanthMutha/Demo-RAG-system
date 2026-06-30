"""
Judges for evaluating RAG output quality.
Reuses prompts from prompt_builder.py.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import RetrievalResult

from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import build_faithfulness_check_prompt


def check_answer_correctness(expected: str, generated: str) -> bool:
    """Check answer correctness via simple string similarity.

    Uses a lightweight approach: normalized text comparison.
    For production, replace with LLM judge or semantic similarity.

    Returns:
        True if generated answer contains the expected answer (case-insensitive),
        or if the expected answer is a substring of the generated answer.
    """
    expected_norm = expected.strip().lower()
    generated_norm = generated.strip().lower()

    if not expected_norm:
        return False

    # Direct containment check
    if expected_norm in generated_norm or generated_norm in expected_norm:
        return True

    # Token overlap ratio
    expected_tokens = set(expected_norm.split())
    generated_tokens = set(generated_norm.split())

    if not expected_tokens:
        return False

    overlap = len(expected_tokens & generated_tokens)
    return overlap / len(expected_tokens) >= 0.5


async def check_faithfulness_with_llm(context: str, answer: str) -> bool:
    """
    Check if an answer is faithful to the given context using an LLM judge.

    Reuses build_faithfulness_check_prompt from prompt_builder.py.
    Returns True if the LLM responds YES, False otherwise.
    """
    prompt = build_faithfulness_check_prompt(context, answer)
    llm = LLMClient()
    try:
        response, _, _ = await llm.generate(prompt=prompt, system_prompt="")
    except Exception:
        return False

    cleaned = response.strip().upper()
    return "YES" in cleaned


async def check_faithfulness(
    context_chunks: list, answer: str, method: str = "heuristic"
) -> bool:
    """
    Check faithfulness using the specified method.

    Args:
        context_chunks: List of RetrievalResult or strings.
        answer: Generated answer text.
        method: "llm" or "heuristic".

    Returns:
        True if the answer appears faithful to the context.
    """
    if method == "llm":
        context_text = "\n\n".join(
            c.content if hasattr(c, "content") else str(c) for c in context_chunks
        )
        return await check_faithfulness_with_llm(context_text, answer)

    # Heuristic: check for significant token overlap
    context_text = " ".join(
        c.content if hasattr(c, "content") else str(c) for c in context_chunks
    ).lower()
    answer_lower = answer.lower()

    if not context_text or not answer_lower:
        return False

    answer_tokens = set(answer_lower.split())
    context_tokens = set(context_text.split())

    if not answer_tokens:
        return False

    overlap = len(answer_tokens & context_tokens)
    return overlap / len(answer_tokens) >= 0.3
