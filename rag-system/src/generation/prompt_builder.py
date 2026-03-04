"""
Prompt builder: constructs RAG prompts with numbered citations and context fitting.
"""
from typing import List

from src.models import RetrievalResult

SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base. \
Answer questions based ONLY on the provided context. If the context does not contain \
enough information to answer, say so clearly. Always cite your sources using [1], [2], etc. \
at the end of each relevant sentence."""


def build_rag_prompt(
    query: str,
    retrieved_chunks: List[RetrievalResult],
) -> str:
    """
    Construct a RAG prompt with numbered context chunks.

    Format:
        CONTEXT:
        [1] {chunk_1_content}
        Source: {source_1}

        [2] {chunk_2_content}
        Source: {source_2}

        QUESTION: {query}

        ANSWER:

    Args:
        query: User's question.
        retrieved_chunks: Ordered list of relevant chunks (highest score first).

    Returns:
        Formatted prompt string ready to send to LLM.
    """
    context_parts: List[str] = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[{idx}] {chunk.content}\nSource: {chunk.document_source}"
        )

    context_block = "\n\n".join(context_parts)

    prompt = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )
    return prompt


def build_faithfulness_check_prompt(context: str, answer: str) -> str:
    """
    Build a prompt for an LLM judge to check answer faithfulness.

    Returns:
        Prompt asking model to respond YES or NO.
    """
    return (
        "Given the following context and answer, determine if the answer "
        "is FULLY supported by the context with no hallucinations.\n"
        "Answer with exactly YES or NO.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Is the answer fully supported by the context? (YES/NO):"
    )
