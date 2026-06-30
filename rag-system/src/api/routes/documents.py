"""
Document listing API routes.
"""
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.database.repository import DocumentRepository

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.get("", summary="List ingested documents")
async def list_documents(
    session: AsyncSession = Depends(get_session),
) -> List[Dict[str, Any]]:
    """Return ingested documents with chunk counts."""
    repo = DocumentRepository(session)
    documents = await repo.list_documents(limit=50)
    return [
        {
            **document,
            "created_at": document["created_at"].isoformat() if document["created_at"] else None,
        }
        for document in documents
    ]
