"""
Database initialization script.
Run once to set up PostgreSQL database and tables.

Usage:
    python scripts/setup_database.py
"""
import asyncio
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.database import init_db
from src.config import settings


async def main() -> None:
    logger.info("Setting up database", url=settings.database_url)
    try:
        await init_db()
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error("❌ Database setup failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
