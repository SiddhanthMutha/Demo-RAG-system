# Product Requirements Document (PRD)
# Multi-Source RAG System with Evaluation Pipeline

**Version:** 1.0  
**Last Updated:** 2026-03-04  
**Document Owner:** Technical Product  
**Target Audience:** AI Implementation Agent / Development Team

---

## Executive Summary

Build a production-quality Retrieval-Augmented Generation (RAG) system that demonstrates enterprise-grade capabilities including multi-format document ingestion, hybrid search retrieval, streaming responses, and automated quality evaluation. This system will serve as a portfolio piece showcasing deep understanding of the complete RAG lifecycle, not just API integration.

**Core Value Proposition:** A RAG system that any AI agent can implement following this specification to create a production-ready, evaluated, and deployable solution.

---

## 1. Project Overview

### 1.1 Objectives

**Primary Goals:**
1. Ingest and process multiple document formats (PDF, web pages, code repositories, markdown)
2. Provide semantic search with high-quality retrieval using hybrid search techniques
3. Generate streaming responses with proper citations and context management
4. Include automated evaluation framework with quantitative metrics
5. Deploy as a containerized REST API with CI/CD pipeline

**Success Criteria:**
- Retrieval precision > 0.80 on golden test dataset
- Average query latency < 1000ms (excluding LLM generation time)
- API uptime > 99% in local testing
- Cost per query < $0.005
- 100% test coverage on core retrieval logic
- Complete documentation enabling others to run and extend the system

### 1.2 Scope

**In Scope:**
- Document ingestion pipeline supporting PDF, HTML, Markdown, and Python/JavaScript code
- Vector database integration with metadata filtering
- Hybrid retrieval (semantic + keyword search)
- Reranking pipeline with cross-encoder
- Streaming API with WebSocket support
- Token budget management and context window optimization
- Automated evaluation framework with golden dataset
- Docker containerization and basic deployment
- Cost tracking and performance monitoring
- CI/CD pipeline with GitHub Actions

**Out of Scope (Future Iterations):**
- User authentication and multi-tenancy
- Fine-tuned embedding models
- Real-time document updates via webhooks
- Kubernetes orchestration (basic manifests OK, full setup not required)
- Advanced caching strategies (Redis integration)
- Multi-language support (English only for v1)

### 1.3 Target Users

**Primary Persona:** Technical recruiters and hiring managers reviewing portfolio projects  
**Secondary Persona:** Developers wanting to understand production RAG implementation

---

## 2. Technical Architecture

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG System                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Ingestion  │──────│   Storage    │──────│  Retrieval   │  │
│  │   Pipeline   │      │   Layer      │      │   Engine     │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                      │          │
│         │                      │                      │          │
│         ▼                      ▼                      ▼          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Generation & Streaming Layer                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│                      ┌───────────────┐                          │
│                      │   FastAPI     │                          │
│                      │   REST API    │                          │
│                      └───────────────┘                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
  ┌───────────┐          ┌───────────┐          ┌───────────┐
  │ Vector DB │          │PostgreSQL │          │   LLM     │
  │ (Pinecone)│          │ (pgvector)│          │   APIs    │
  └───────────┘          └───────────┘          └───────────┘
```

### 2.2 Technology Stack

**Core Framework & Language:**
- Python 3.11+ (required for latest async features and performance)
- Type hints throughout (PEP 484, 526)
- Async/await patterns for I/O operations

**API & Web Framework:**
- FastAPI 0.104+ (OpenAPI documentation, async support)
- Uvicorn with WebSocket support
- Pydantic 2.5+ for validation

**LLM Providers:**
- OpenAI API (GPT-4, GPT-3.5-turbo for embeddings)
- Anthropic API (Claude 3.5 Sonnet for comparison)
- Support for switching between providers via configuration

**Vector Database (Choose One - Pinecone Recommended):**
- **Primary:** Pinecone (serverless, production-ready)
- **Alternative:** Qdrant (self-hosted option)
- Must support: metadata filtering, hybrid search, batch upsert

**Relational Database:**
- PostgreSQL 15+ with pgvector extension
- Used for: document metadata, user queries, evaluation results

**Document Processing:**
- PyPDF2 (PDF extraction)
- BeautifulSoup4 + requests (web scraping)
- python-markdown (markdown parsing)
- tree-sitter (syntax-aware code chunking - optional but impressive)

**Embeddings & Search:**
- sentence-transformers (local embeddings, default: all-MiniLM-L6-v2)
- rank-bm25 (keyword search)
- cross-encoder/ms-marco-MiniLM-L-6-v2 (reranking)

**Testing & Evaluation:**
- pytest + pytest-asyncio (unit and integration tests)
- httpx (async API testing)
- Custom evaluation framework (ragas optional but recommended)

**Infrastructure:**
- Docker + docker-compose
- GitHub Actions (CI/CD)
- Pre-commit hooks (black, flake8, mypy)

**Utilities:**
- tiktoken (token counting)
- tenacity (retry logic with exponential backoff)
- loguru (structured logging)
- python-dotenv (environment management)

### 2.3 Project Directory Structure

```
rag-system/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Run tests on PR
│       ├── docker-build.yml          # Build and push images
│       └── evaluation.yml            # Run RAG evaluation suite
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   ├── models.py                     # Pydantic models
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── base_parser.py            # Abstract parser interface
│   │   ├── pdf_parser.py             # PDF document parser
│   │   ├── web_parser.py             # Web scraping parser
│   │   ├── code_parser.py            # Code file parser
│   │   ├── markdown_parser.py        # Markdown parser
│   │   └── chunker.py                # Chunking strategies
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py             # Embedding service
│   │   ├── vector_store.py           # Vector DB interface
│   │   ├── keyword_search.py         # BM25 implementation
│   │   ├── hybrid_retriever.py       # Combined retrieval
│   │   └── reranker.py               # Cross-encoder reranking
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_client.py             # LLM API client
│   │   ├── prompt_builder.py         # Prompt construction
│   │   ├── context_manager.py        # Token budget management
│   │   └── streaming.py              # Streaming response handler
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI application
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── ingest.py             # Document ingestion endpoints
│   │   │   ├── query.py              # Query and chat endpoints
│   │   │   └── health.py             # Health check endpoints
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── logging.py            # Request logging
│   │       └── error_handling.py     # Error handlers
│   │
│   └── database/
│       ├── __init__.py
│       ├── models.py                 # SQLAlchemy models
│       ├── repository.py             # Database operations
│       └── migrations/               # Alembic migrations
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_embeddings.py
│   │   ├── test_retrieval.py
│   │   └── test_prompt_builder.py
│   ├── integration/
│   │   ├── test_api_endpoints.py
│   │   ├── test_ingestion_pipeline.py
│   │   └── test_end_to_end.py
│   └── evaluation/
│       ├── golden_dataset.json       # Ground truth Q&A pairs
│       ├── test_retrieval_quality.py # Retrieval metrics
│       ├── test_generation_quality.py # Answer quality metrics
│       └── test_performance.py       # Latency and cost tests
│
├── scripts/
│   ├── ingest_documents.py           # Bulk document ingestion
│   ├── run_evaluation.py             # Run evaluation suite
│   ├── setup_database.py             # Database initialization
│   ├── benchmark_embeddings.py       # Compare embedding models
│   └── cost_analysis.py              # Generate cost reports
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile                # Application container
│   │   ├── Dockerfile.dev            # Development container
│   │   └── docker-compose.yml        # Multi-container setup
│   └── k8s/                          # Optional Kubernetes manifests
│       ├── deployment.yml
│       ├── service.yml
│       └── configmap.yml
│
├── data/
│   ├── raw/                          # Unprocessed documents
│   ├── processed/                    # Chunked and embedded
│   └── evaluation/                   # Test datasets
│
├── docs/
│   ├── architecture.md               # System design documentation
│   ├── api_reference.md              # API documentation
│   ├── evaluation_results.md         # Performance benchmarks
│   └── deployment_guide.md           # Deployment instructions
│
├── .env.example                      # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml          # Pre-commit hooks
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies
├── pytest.ini                        # Pytest configuration
├── mypy.ini                          # Type checking configuration
├── README.md                         # Project README
└── LICENSE                           # MIT License

```

---

## 3. Detailed Component Specifications

### 3.1 Data Models

#### 3.1.1 Core Models (src/models.py)

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    WEB = "web"
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"

class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SYNTAX_AWARE = "syntax_aware"  # For code

class Document(BaseModel):
    """Raw document before processing"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., min_length=1)
    doc_type: DocumentType
    source: str = Field(..., description="File path, URL, or identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Machine learning is...",
                "doc_type": "pdf",
                "source": "/data/ml_paper.pdf",
                "metadata": {"author": "John Doe", "pages": 12}
            }
        }

class Chunk(BaseModel):
    """Processed chunk ready for embedding"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str = Field(..., min_length=1)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    
    @validator('embedding')
    def validate_embedding_dimension(cls, v):
        if v is not None and len(v) not in [384, 768, 1536]:  # Common dimensions
            raise ValueError("Embedding dimension must be 384, 768, or 1536")
        return v

class QueryRequest(BaseModel):
    """User query request"""
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None
    use_reranking: bool = Field(default=True)
    stream: bool = Field(default=True)
    model: Literal["gpt-4", "gpt-3.5-turbo", "claude-3-5-sonnet"] = "gpt-3.5-turbo"

class RetrievalResult(BaseModel):
    """Single retrieval result"""
    chunk_id: str
    content: str
    score: float = Field(..., ge=0, le=1)
    metadata: Dict[str, Any]
    document_source: str

class QueryResponse(BaseModel):
    """Complete query response"""
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    answer: str
    sources: List[RetrievalResult]
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "tokens_used": 0,
            "latency_ms": 0,
            "cost_usd": 0.0
        }
    )
    
class IngestionRequest(BaseModel):
    """Document ingestion request"""
    source: str = Field(..., description="File path or URL")
    doc_type: DocumentType
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    metadata: Optional[Dict[str, Any]] = None

class IngestionResponse(BaseModel):
    """Ingestion result"""
    document_id: str
    chunks_created: int
    status: Literal["success", "partial", "failed"]
    errors: List[str] = Field(default_factory=list)
```

#### 3.1.2 Database Models (src/database/models.py)

```python
from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class DocumentRecord(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    content_hash = Column(String, unique=True, index=True)
    doc_type = Column(String, nullable=False)
    source = Column(String, nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chunks = relationship("ChunkRecord", back_populates="document", cascade="all, delete-orphan")

class ChunkRecord(Base):
    __tablename__ = "chunks"
    
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # Adjust dimension based on model
    token_count = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    metadata = Column(JSON, default={})
    
    document = relationship("DocumentRecord", back_populates="chunks")

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(String, primary_key=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    retrieval_results = Column(JSON)  # Store chunk IDs and scores
    latency_ms = Column(Float)
    tokens_used = Column(Integer)
    cost_usd = Column(Float)
    model_used = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    
    id = Column(String, primary_key=True)
    test_name = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    details = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 3.2 Ingestion Pipeline

#### 3.2.1 Parser Interface (src/ingestion/base_parser.py)

**Requirements:**
- Abstract base class defining parser contract
- All parsers must implement `parse()` method returning `Document`
- Error handling with specific exception types
- Support for batch processing

**Implementation Details:**

```python
from abc import ABC, abstractmethod
from typing import List
from ..models import Document, DocumentType

class ParsingError(Exception):
    """Custom exception for parsing errors"""
    pass

class BaseParser(ABC):
    """Abstract base class for document parsers"""
    
    @property
    @abstractmethod
    def supported_types(self) -> List[DocumentType]:
        """Return list of supported document types"""
        pass
    
    @abstractmethod
    async def parse(self, source: str, **kwargs) -> Document:
        """
        Parse document from source into Document object.
        
        Args:
            source: File path, URL, or raw content
            **kwargs: Parser-specific options
            
        Returns:
            Document object with extracted content
            
        Raises:
            ParsingError: If parsing fails
        """
        pass
    
    async def parse_batch(self, sources: List[str], **kwargs) -> List[Document]:
        """Parse multiple documents concurrently"""
        import asyncio
        tasks = [self.parse(source, **kwargs) for source in sources]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

#### 3.2.2 PDF Parser (src/ingestion/pdf_parser.py)

**Requirements:**
- Extract text from PDF files
- Preserve page numbers in metadata
- Handle multi-column layouts
- Extract embedded images (metadata only, not content)
- Support password-protected PDFs (with password parameter)

**Key Features:**
- Use PyPDF2 as primary library
- Fall back to pdfplumber for complex layouts
- Store metadata: page count, author, title, creation date
- Handle corrupted PDFs gracefully

#### 3.2.3 Web Parser (src/ingestion/web_parser.py)

**Requirements:**
- Fetch HTML from URL
- Extract main content (ignore navigation, ads, footers)
- Convert HTML to clean markdown
- Respect robots.txt
- Handle JavaScript-rendered content (optional: use playwright)

**Key Features:**
- Use requests + BeautifulSoup4
- Implement rate limiting (1 request per second)
- Extract metadata: title, author, publish date (from meta tags)
- Follow redirects (max 3)
- User-Agent header: "RAGSystemBot/1.0"

#### 3.2.4 Code Parser (src/ingestion/code_parser.py)

**Requirements:**
- Parse Python and JavaScript files
- Syntax-aware chunking (keep functions/classes intact)
- Extract docstrings and comments
- Preserve import statements in metadata

**Key Features:**
- Use tree-sitter for AST parsing
- Chunk by: functions, classes, top-level blocks
- Store metadata: file path, language, LOC
- Handle syntax errors gracefully

#### 3.2.5 Chunking Strategy (src/ingestion/chunker.py)

**Requirements:**
- Support multiple chunking strategies
- Configurable chunk size and overlap
- Token counting with tiktoken
- Preserve semantic boundaries

**Chunking Strategies:**

1. **Fixed-Size Chunking:**
   - Default: 512 tokens per chunk
   - Overlap: 50 tokens
   - Use for: code, structured data

2. **Semantic Chunking:**
   - Split on sentence boundaries
   - Combine sentences until token limit
   - Use for: prose, articles, documentation

3. **Syntax-Aware Chunking (Code):**
   - One function/class per chunk
   - Max 1000 tokens (split large functions)
   - Include docstring with function

**Implementation Requirements:**

```python
from typing import List, Protocol
from ..models import Document, Chunk, ChunkingStrategy
import tiktoken

class Chunker:
    def __init__(
        self,
        strategy: ChunkingStrategy,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        model_name: str = "gpt-3.5-turbo"
    ):
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = tiktoken.encoding_for_model(model_name)
    
    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        Split document into chunks based on strategy.
        
        Returns:
            List of Chunk objects with metadata
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunk(doc)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunk(doc)
        elif self.strategy == ChunkingStrategy.SYNTAX_AWARE:
            return self._syntax_aware_chunk(doc)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
    
    # Implement each strategy...
```

### 3.3 Retrieval System

#### 3.3.1 Embedding Service (src/retrieval/embeddings.py)

**Requirements:**
- Generate embeddings for text
- Support batch processing
- Cache embeddings to avoid recomputation
- Multiple model support

**Models to Support:**
- Local: `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fast)
- OpenAI: `text-embedding-3-small` (1536 dim, high quality)
- Optional: `text-embedding-ada-002` (1536 dim, legacy)

**Implementation Requirements:**

```python
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import openai
from functools import lru_cache

class EmbeddingService:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_openai: bool = False,
        cache_size: int = 1000
    ):
        self.use_openai = use_openai
        if use_openai:
            self.model_name = "text-embedding-3-small"
        else:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
    
    @lru_cache(maxsize=1000)
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        pass
    
    async def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for batch of texts"""
        pass
    
    def get_dimension(self) -> int:
        """Return embedding dimension"""
        pass
```

**Performance Requirements:**
- Batch processing: >100 embeddings/second (local model)
- Memory efficient: stream large batches
- Error handling: retry on API failures with exponential backoff

#### 3.3.2 Vector Store (src/retrieval/vector_store.py)

**Requirements:**
- Abstract interface supporting multiple vector DBs
- CRUD operations for chunks
- Similarity search with metadata filtering
- Batch upsert for efficiency

**Pinecone Implementation:**

```python
from typing import List, Optional, Dict, Any
import pinecone
from ..models import Chunk, RetrievalResult

class VectorStore:
    """Vector database interface"""
    
    def __init__(self, config: Dict[str, str]):
        """
        Initialize vector store.
        
        Args:
            config: {
                'api_key': str,
                'environment': str,
                'index_name': str
            }
        """
        pass
    
    async def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """
        Insert or update chunks in vector DB.
        
        Args:
            chunks: List of Chunk objects with embeddings
            
        Returns:
            Number of successfully inserted chunks
        """
        pass
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Semantic search for similar chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {'doc_type': 'pdf'})
            
        Returns:
            List of retrieval results sorted by score
        """
        pass
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics (total vectors, dimension, etc.)"""
        pass
```

**Metadata Filtering Requirements:**
- Support exact match: `{"doc_type": "pdf"}`
- Support range queries: `{"created_at": {"$gte": "2024-01-01"}}`
- Support multiple filters: `{"doc_type": "pdf", "author": "Smith"}`

#### 3.3.3 Keyword Search (src/retrieval/keyword_search.py)

**Requirements:**
- BM25 implementation for keyword matching
- Support for boolean queries
- Integration with vector search for hybrid retrieval

**Implementation:**

```python
from rank_bm25 import BM25Okapi
from typing import List, Tuple
from ..models import Chunk

class KeywordSearchEngine:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []
    
    def index_chunks(self, chunks: List[Chunk]):
        """Build BM25 index from chunks"""
        tokenized_corpus = [chunk.content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunks = chunks
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Search using BM25.
        
        Returns:
            List of (chunk, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # Return top-k chunks with scores
```

#### 3.3.4 Hybrid Retrieval (src/retrieval/hybrid_retriever.py)

**Requirements:**
- Combine vector and keyword search results
- Reciprocal Rank Fusion (RRF) for merging
- Configurable weighting between semantic and keyword

**RRF Formula:**
```
score(d) = Σ 1 / (k + rank_i(d))
```
where k=60 (standard constant)

**Implementation:**

```python
from typing import List
from ..models import RetrievalResult

class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_search: KeywordSearchEngine,
        alpha: float = 0.5  # Weight for vector search
    ):
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.alpha = alpha
    
    async def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval.
        
        Algorithm:
        1. Get top_k*2 results from vector search
        2. Get top_k*2 results from keyword search
        3. Merge using RRF
        4. Return top_k combined results
        """
        pass
```

#### 3.3.5 Reranking (src/retrieval/reranker.py)

**Requirements:**
- Cross-encoder model for reranking
- Rerank top candidates from hybrid retrieval
- Compute relevance scores

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Implementation:**

```python
from sentence_transformers import CrossEncoder
from typing import List, Tuple

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results using cross-encoder.
        
        Args:
            query: User query
            results: Initial retrieval results
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked results with updated scores
        """
        # Create query-document pairs
        pairs = [(query, result.content) for result in results]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score and return top_k
```

### 3.4 Generation & Streaming

#### 3.4.1 LLM Client (src/generation/llm_client.py)

**Requirements:**
- Support OpenAI and Anthropic APIs
- Streaming response handling
- Retry logic with exponential backoff
- Token counting and cost tracking

**Supported Models:**
- OpenAI: `gpt-4`, `gpt-3.5-turbo`
- Anthropic: `claude-3-5-sonnet-20241022`

**Implementation:**

```python
from typing import AsyncIterator, Dict, Any
import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMClient:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            self.client = openai.AsyncOpenAI(api_key=api_key)
        elif provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> AsyncIterator[str]:
        """
        Generate streaming response.
        
        Yields:
            Token strings as they are generated
        """
        if self.provider == "openai":
            # OpenAI streaming implementation
            pass
        elif self.provider == "anthropic":
            # Anthropic streaming implementation
            pass
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost in USD.
        
        Pricing (as of 2024):
        - gpt-3.5-turbo: $0.0015/1K input, $0.002/1K output
        - gpt-4: $0.03/1K input, $0.06/1K output
        - claude-3-5-sonnet: $0.003/1K input, $0.015/1K output
        """
        pass
```

#### 3.4.2 Prompt Builder (src/generation/prompt_builder.py)

**Requirements:**
- Construct prompts with retrieved context
- Include citations/source references
- Handle context window limits
- Template system for different query types

**Prompt Template:**

```python
SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base.
Answer questions based on the provided context. If you cannot answer from the context,
say so clearly. Always cite your sources using [1], [2], etc."""

def build_rag_prompt(
    query: str,
    retrieved_chunks: List[RetrievalResult],
    max_context_tokens: int = 3000
) -> str:
    """
    Build prompt with context and query.
    
    Format:
    CONTEXT:
    [1] {chunk_1_content}
    Source: {source_1}
    
    [2] {chunk_2_content}
    Source: {source_2}
    
    QUESTION: {query}
    
    ANSWER:
    """
    pass
```

**Context Window Management:**
- GPT-3.5-turbo: 16K tokens total
- Reserve 1000 tokens for response
- Reserve 500 tokens for system/user prompt
- Use remaining ~14.5K for context
- Truncate context if needed (keep highest-scored chunks)

#### 3.4.3 Context Manager (src/generation/context_manager.py)

**Requirements:**
- Count tokens accurately
- Prioritize chunks by relevance score
- Fit within model's context window
- Track token usage for cost estimation

```python
import tiktoken
from typing import List
from ..models import RetrievalResult

class ContextManager:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.max_tokens = self._get_model_limit(model_name)
    
    def _get_model_limit(self, model_name: str) -> int:
        """Return context window size for model"""
        limits = {
            "gpt-3.5-turbo": 16385,
            "gpt-4": 8192,
            "claude-3-5-sonnet-20241022": 200000
        }
        return limits.get(model_name, 4096)
    
    def fit_context(
        self,
        chunks: List[RetrievalResult],
        system_prompt: str,
        query: str,
        max_response_tokens: int = 1000
    ) -> List[RetrievalResult]:
        """
        Select chunks that fit within context window.
        
        Algorithm:
        1. Count tokens in system prompt + query + response budget
        2. Calculate remaining token budget for context
        3. Sort chunks by score (descending)
        4. Add chunks until budget exhausted
        5. Return selected chunks
        """
        pass
```

### 3.5 API Layer

#### 3.5.1 FastAPI Application (src/api/main.py)

**Requirements:**
- RESTful API with OpenAPI docs
- WebSocket support for streaming
- CORS enabled for frontend integration
- Request/response logging
- Error handling middleware

**Endpoints:**

```python
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ..models import QueryRequest, QueryResponse, IngestionRequest

app = FastAPI(
    title="RAG System API",
    description="Production-quality RAG system with multi-source ingestion",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/v1/ingest", response_model=IngestionResponse)
async def ingest_document(request: IngestionRequest):
    """
    Ingest a document into the system.
    
    Steps:
    1. Parse document based on type
    2. Chunk content
    3. Generate embeddings
    4. Store in vector DB and PostgreSQL
    """
    pass

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system (non-streaming).
    
    Steps:
    1. Generate query embedding
    2. Retrieve relevant chunks (hybrid search)
    3. Rerank results
    4. Build prompt with context
    5. Generate response
    6. Return with sources
    """
    pass

@app.websocket("/api/v1/query/stream")
async def query_stream(websocket: WebSocket):
    """
    Streaming query endpoint.
    
    Protocol:
    - Client sends: {"query": str, "top_k": int}
    - Server sends: {"type": "token", "data": str} for each token
    - Server sends: {"type": "sources", "data": [...]} at end
    - Server sends: {"type": "done"} when complete
    """
    await websocket.accept()
    # Implementation
```

#### 3.5.2 Route Handlers (src/api/routes/)

**Ingestion Route (ingest.py):**
- File upload support (multipart/form-data)
- URL ingestion support
- Batch ingestion endpoint
- Status tracking for long-running ingestions

**Query Route (query.py):**
- Sync and async query endpoints
- WebSocket streaming
- Query history endpoint
- Feedback submission (for evaluation)

**Health Route (health.py):**
- Basic health check
- Dependency health (DB, vector store, LLM API)
- System metrics (requests/sec, avg latency)

#### 3.5.3 Middleware (src/api/middleware/)

**Logging Middleware (logging.py):**
- Log all requests with UUID
- Track latency per endpoint
- Log errors with full context
- Use structured logging (JSON format)

**Error Handling (error_handling.py):**
- Custom exception handlers
- Return consistent error format
- Mask sensitive information
- Log errors to monitoring system

### 3.6 Evaluation Framework

#### 3.6.1 Golden Dataset (tests/evaluation/golden_dataset.json)

**Requirements:**
- 50-100 question-answer pairs with ground truth
- Cover diverse query types
- Include expected source documents
- Regular updates as system evolves

**Format:**

```json
[
  {
    "id": "eval_001",
    "query": "What are the key features of transformers?",
    "expected_answer": "Transformers use self-attention mechanisms...",
    "expected_sources": ["doc_123", "doc_456"],
    "expected_chunks": ["chunk_789", "chunk_012"],
    "category": "technical_explanation",
    "difficulty": "medium"
  },
  {
    "id": "eval_002",
    "query": "Who invented the transformer architecture?",
    "expected_answer": "Vaswani et al. from Google...",
    "expected_sources": ["doc_attention_paper"],
    "expected_chunks": ["chunk_intro"],
    "category": "factual",
    "difficulty": "easy"
  }
]
```

#### 3.6.2 Retrieval Quality Tests (tests/evaluation/test_retrieval_quality.py)

**Metrics to Measure:**

1. **Precision@K:**
   ```
   Precision@K = (Relevant chunks retrieved in top K) / K
   ```
   Target: >0.80 for K=5

2. **Recall@K:**
   ```
   Recall@K = (Relevant chunks retrieved) / (Total relevant chunks)
   ```
   Target: >0.70 for K=10

3. **MRR (Mean Reciprocal Rank):**
   ```
   MRR = Average(1 / rank of first relevant result)
   ```
   Target: >0.75

4. **NDCG@K (Normalized Discounted Cumulative Gain):**
   ```
   NDCG@K = DCG@K / IDCG@K
   ```
   Target: >0.80

**Implementation:**

```python
import pytest
from typing import List, Set
import json

@pytest.fixture
def golden_dataset():
    with open("tests/evaluation/golden_dataset.json") as f:
        return json.load(f)

def calculate_precision_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int
) -> float:
    """Calculate precision@k"""
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / k

@pytest.mark.asyncio
async def test_retrieval_precision(golden_dataset, retriever):
    """Test retrieval precision across golden dataset"""
    precisions = []
    
    for item in golden_dataset:
        query = item["query"]
        expected_chunks = set(item["expected_chunks"])
        
        # Retrieve chunks
        results = await retriever.retrieve(query, top_k=5)
        retrieved_ids = [r.chunk_id for r in results]
        
        # Calculate precision
        precision = calculate_precision_at_k(retrieved_ids, expected_chunks, k=5)
        precisions.append(precision)
    
    avg_precision = sum(precisions) / len(precisions)
    assert avg_precision > 0.80, f"Precision@5: {avg_precision:.3f} < 0.80"
```

#### 3.6.3 Generation Quality Tests (tests/evaluation/test_generation_quality.py)

**Metrics to Measure:**

1. **Answer Relevance:**
   - Use LLM-as-judge to score answer relevance (1-5 scale)
   - Compare generated answer to expected answer
   - Target: >4.0 average

2. **Faithfulness:**
   - Verify answer is grounded in retrieved context
   - No hallucinations
   - Use LLM to check if answer is supported by sources
   - Target: >0.90 (90% of answers are faithful)

3. **Citation Accuracy:**
   - Check if cited sources are actually used
   - Verify citation numbers match provided sources
   - Target: 100% accurate citations

**Implementation:**

```python
@pytest.mark.asyncio
async def test_answer_faithfulness(golden_dataset, rag_system):
    """Test if answers are grounded in sources"""
    
    faithfulness_scores = []
    
    for item in golden_dataset:
        query = item["query"]
        response = await rag_system.query(query)
        
        # Use LLM to judge faithfulness
        prompt = f"""
        Given the following context and answer, determine if the answer
        is fully supported by the context. Answer with YES or NO.
        
        CONTEXT:
        {response.sources}
        
        ANSWER:
        {response.answer}
        """
        
        judgment = await judge_llm.generate(prompt)
        faithfulness_scores.append(1 if "YES" in judgment else 0)
    
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    assert avg_faithfulness > 0.90
```

#### 3.6.4 Performance Tests (tests/evaluation/test_performance.py)

**Metrics:**

1. **Latency:**
   - End-to-end query latency (p50, p95, p99)
   - Target: p95 < 1000ms (excluding LLM generation)

2. **Cost:**
   - Cost per query (embedding + LLM tokens)
   - Target: <$0.005 per query

3. **Throughput:**
   - Queries per second
   - Target: >10 QPS on single instance

**Implementation:**

```python
import time
import statistics

@pytest.mark.asyncio
async def test_query_latency(rag_system):
    """Measure query latency"""
    
    test_queries = ["What is machine learning?"] * 100
    latencies = []
    
    for query in test_queries:
        start = time.time()
        await rag_system.query(query)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
    
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=100)[94]
    p99 = statistics.quantiles(latencies, n=100)[98]
    
    print(f"Latency - p50: {p50:.0f}ms, p95: {p95:.0f}ms, p99: {p99:.0f}ms")
    assert p95 < 1000, f"p95 latency {p95:.0f}ms exceeds 1000ms"

@pytest.mark.asyncio
async def test_cost_per_query(rag_system, golden_dataset):
    """Measure cost per query"""
    
    total_cost = 0.0
    
    for item in golden_dataset:
        response = await rag_system.query(item["query"])
        total_cost += response.metadata["cost_usd"]
    
    avg_cost = total_cost / len(golden_dataset)
    
    print(f"Average cost per query: ${avg_cost:.5f}")
    assert avg_cost < 0.005
```

---

## 4. Configuration Management

### 4.1 Environment Variables (.env.example)

```bash
# LLM APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Vector Database (Pinecone)
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=rag-system

# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/ragdb

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2  # or 'openai'
EMBEDDING_DIMENSION=384

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# RAG Configuration
DEFAULT_TOP_K=5
MAX_CONTEXT_TOKENS=3000
DEFAULT_CHUNK_SIZE=512
DEFAULT_CHUNK_OVERLAP=50

# Feature Flags
USE_RERANKING=true
USE_HYBRID_SEARCH=true
ENABLE_QUERY_LOGGING=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### 4.2 Configuration Class (src/config.py)

```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM APIs
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    
    # Vector DB
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "rag-system"
    
    # Database
    database_url: str
    
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # RAG
    default_top_k: int = 5
    max_context_tokens: int = 3000
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50
    
    # Feature flags
    use_reranking: bool = True
    use_hybrid_search: bool = True
    enable_query_logging: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 5. Testing Strategy

### 5.1 Unit Tests (tests/unit/)

**Coverage Target:** >80% line coverage

**Test Files:**
- `test_chunker.py`: Test chunking strategies
- `test_embeddings.py`: Test embedding generation
- `test_prompt_builder.py`: Test prompt construction
- `test_context_manager.py`: Test token counting
- `test_parsers.py`: Test each document parser

**Example Test:**

```python
import pytest
from src.ingestion.chunker import Chunker, ChunkingStrategy
from src.models import Document, DocumentType

def test_fixed_size_chunking():
    chunker = Chunker(
        strategy=ChunkingStrategy.FIXED_SIZE,
        max_tokens=100,
        overlap_tokens=10
    )
    
    doc = Document(
        content="This is a test. " * 200,  # Long text
        doc_type=DocumentType.TEXT,
        source="test.txt"
    )
    
    chunks = chunker.chunk_document(doc)
    
    # Assertions
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
```

### 5.2 Integration Tests (tests/integration/)

**Test Scenarios:**
- Full ingestion pipeline (document → chunks → embeddings → vector DB)
- End-to-end query (query → retrieval → generation → response)
- API endpoint tests (ingest, query, health)

**Example:**

```python
@pytest.mark.asyncio
async def test_end_to_end_query(test_client, sample_documents):
    # 1. Ingest documents
    for doc in sample_documents:
        response = await test_client.post("/api/v1/ingest", json=doc)
        assert response.status_code == 200
    
    # 2. Query
    query_request = {
        "query": "What is machine learning?",
        "top_k": 5
    }
    response = await test_client.post("/api/v1/query", json=query_request)
    
    # 3. Validate response
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) <= 5
    assert data["metadata"]["tokens_used"] > 0
```

### 5.3 Evaluation Tests (tests/evaluation/)

- Already covered in section 3.6

### 5.4 Test Fixtures (tests/conftest.py)

```python
import pytest
import asyncio
from src.api.main import app
from httpx import AsyncClient

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def sample_documents():
    return [
        {
            "source": "test_doc_1.txt",
            "doc_type": "text",
            "content": "Machine learning is a subset of AI..."
        },
        # More sample docs...
    ]

@pytest.fixture
async def populated_vector_store():
    # Setup: Create and populate vector store
    # Yield store
    # Teardown: Clean up
    pass
```

---

## 6. Deployment & Infrastructure

### 6.1 Docker Configuration

#### Dockerfile (infra/docker/Dockerfile)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml (infra/docker/docker-compose.yml)

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ragpassword
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U raguser"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build:
      context: ../..
      dockerfile: infra/docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://raguser:ragpassword@postgres:5432/ragdb
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      PINECONE_API_KEY: ${PINECONE_API_KEY}
      PINECONE_ENVIRONMENT: ${PINECONE_ENVIRONMENT}
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./data:/app/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### 6.2 CI/CD Pipeline

#### GitHub Actions Workflow (.github/workflows/ci.yml)

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: pgvector/pgvector:pg15
        env:
          POSTGRES_PASSWORD: testpassword
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linters
      run: |
        black --check src/ tests/
        flake8 src/ tests/
        mypy src/
    
    - name: Run unit tests
      run: pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://postgres:testpassword@localhost:5432/testdb
      run: pytest tests/integration/ -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml

  evaluation:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'pull_request'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Run evaluation suite
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
      run: python scripts/run_evaluation.py
    
    - name: Comment results on PR
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const results = fs.readFileSync('evaluation_results.md', 'utf8');
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: results
          });
```

---

## 7. Documentation Requirements

### 7.1 README.md Structure

```markdown
# Multi-Source RAG System

Production-quality RAG system with multi-format ingestion and automated evaluation.

## Features
- ✅ Multi-format document ingestion (PDF, web, code, markdown)
- ✅ Hybrid retrieval (semantic + keyword search)
- ✅ Cross-encoder reranking
- ✅ Streaming API with WebSocket support
- ✅ Automated evaluation framework
- ✅ Token budget management
- ✅ Cost tracking

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI API key
- Pinecone account

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start with Docker Compose
docker-compose up -d
```

### Usage

**Ingest a document:**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/path/to/document.pdf",
    "doc_type": "pdf"
  }'
```

**Query the system:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5
  }'
```

## Architecture

[Insert architecture diagram]

## Performance Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Retrieval Precision@5 | >0.80 | 0.84 |
| Answer Faithfulness | >0.90 | 0.92 |
| P95 Latency | <1000ms | 850ms |
| Cost per Query | <$0.005 | $0.003 |

## Design Decisions

### Why Pinecone over FAISS?
- Serverless, no infrastructure management
- Built-in metadata filtering
- Better for production scalability
- Trade-off: Cost vs self-hosted FAISS

### Why Hybrid Search?
- Vector search alone misses exact keyword matches
- BM25 provides complementary signal
- RRF fusion improves overall recall

## Development

**Run tests:**
```bash
pytest tests/ -v --cov=src
```

**Run evaluation:**
```bash
python scripts/run_evaluation.py
```

**Code quality:**
```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Known Limitations

1. Single-language support (English only)
2. No real-time document updates
3. Limited to text-based content (no image understanding)
4. Maximum document size: 10MB

## Future Improvements

- [ ] Fine-tuned embedding models
- [ ] Multi-language support
- [ ] Image and table understanding
- [ ] Advanced caching with Redis
- [ ] User feedback loop for continuous improvement

## License

MIT License
```

### 7.2 API Documentation (docs/api_reference.md)

- Auto-generated from FastAPI (OpenAPI spec)
- Manual additions for complex flows
- Example requests/responses
- Error codes and meanings

### 7.3 Architecture Documentation (docs/architecture.md)

- System architecture diagram
- Component interaction flows
- Data flow diagrams
- Technology choices and rationale

### 7.4 Deployment Guide (docs/deployment_guide.md)

- Local development setup
- Docker deployment
- Environment configuration
- Monitoring and logging setup
- Troubleshooting guide

---

## 8. Evaluation & Success Metrics

### 8.1 Quantitative Metrics

**Retrieval Quality:**
- Precision@5: >0.80
- Recall@10: >0.70
- MRR: >0.75
- NDCG@10: >0.80

**Generation Quality:**
- Answer Relevance (1-5): >4.0
- Faithfulness: >0.90
- Citation Accuracy: 100%

**Performance:**
- P50 Latency: <500ms
- P95 Latency: <1000ms
- P99 Latency: <2000ms
- Throughput: >10 QPS

**Cost:**
- Average cost per query: <$0.005
- Daily cost for 1000 queries: <$5

### 8.2 Qualitative Assessment

**Code Quality:**
- Type hints throughout
- Comprehensive docstrings
- Clean separation of concerns
- Following SOLID principles

**Documentation:**
- Clear README with quick start
- Architecture diagrams
- API documentation
- Deployment guide

**Production Readiness:**
- Error handling and retries
- Logging and monitoring
- Health checks
- Graceful degradation

### 8.3 Evaluation Script Output Format

```markdown
# RAG System Evaluation Report

**Date:** 2024-01-15
**Dataset:** golden_dataset.json (50 examples)

## Retrieval Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Precision@5 | 0.84 | >0.80 | ✅ PASS |
| Recall@10 | 0.72 | >0.70 | ✅ PASS |
| MRR | 0.78 | >0.75 | ✅ PASS |
| NDCG@10 | 0.82 | >0.80 | ✅ PASS |

## Generation Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Faithfulness | 0.92 | >0.90 | ✅ PASS |
| Citation Accuracy | 1.00 | 1.00 | ✅ PASS |

## Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| P50 Latency | 420ms | <500ms | ✅ PASS |
| P95 Latency | 850ms | <1000ms | ✅ PASS |
| Avg Cost | $0.0032 | <$0.005 | ✅ PASS |

## Summary

**Overall Status:** ✅ ALL CHECKS PASSED

All metrics meet or exceed target thresholds. System is production-ready.
```

---

## 9. Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal:** Basic working system

**Tasks:**
1. Project setup and structure
2. Core data models (Pydantic)
3. Basic PDF parser
4. Fixed-size chunking
5. OpenAI embeddings integration
6. Pinecone setup and vector storage
7. Simple retrieval (vector search only)
8. Basic prompt construction
9. FastAPI skeleton with query endpoint

**Deliverable:** Can ingest a PDF and answer questions

### Phase 2: Multi-Source & Hybrid Retrieval (Week 2)
**Goal:** Production-quality retrieval

**Tasks:**
1. Web scraper parser
2. Code parser with tree-sitter
3. Markdown parser
4. Semantic chunking strategy
5. BM25 keyword search
6. Hybrid retrieval with RRF
7. Cross-encoder reranking
8. Context window management
9. PostgreSQL integration

**Deliverable:** Multi-format ingestion with high-quality retrieval

### Phase 3: Generation & API (Week 3)
**Goal:** Complete API with streaming

**Tasks:**
1. Anthropic API integration
2. Streaming response handler
3. Token counting and cost tracking
4. WebSocket endpoint
5. Ingestion API endpoints
6. Middleware (logging, error handling)
7. API documentation

**Deliverable:** Full REST API with streaming support

### Phase 4: Evaluation & Testing (Week 4)
**Goal:** Automated quality assurance

**Tasks:**
1. Create golden dataset (50 examples)
2. Retrieval quality tests
3. Generation quality tests
4. Performance benchmarks
5. Unit test coverage
6. Integration tests
7. CI/CD pipeline setup
8. Evaluation script

**Deliverable:** Comprehensive test suite with metrics

### Phase 5: Deployment & Documentation (Week 5)
**Goal:** Production-ready deployment

**Tasks:**
1. Docker containerization
2. Docker Compose setup
3. GitHub Actions workflows
4. README with architecture diagram
5. API reference documentation
6. Deployment guide
7. Cost analysis script
8. Final polish and code cleanup

**Deliverable:** Deployable system with excellent documentation

---

## 10. Implementation Guidelines for AI Agent

### 10.1 Code Standards

**Python Style:**
- Follow PEP 8
- Use type hints for all functions
- Docstrings in Google style format
- Max line length: 100 characters

**Example:**

```python
from typing import List, Optional

async def retrieve_chunks(
    query: str,
    top_k: int = 5,
    filters: Optional[dict] = None
) -> List[RetrievalResult]:
    """
    Retrieve relevant chunks for a query.
    
    Args:
        query: User query string
        top_k: Number of results to return
        filters: Optional metadata filters
        
    Returns:
        List of retrieval results sorted by relevance
        
    Raises:
        ValueError: If top_k < 1
        VectorStoreError: If vector search fails
        
    Example:
        >>> results = await retrieve_chunks("What is AI?", top_k=3)
        >>> len(results)
        3
    """
    pass
```

**Error Handling:**
- Use custom exception classes
- Always include context in error messages
- Log errors with full traceback
- Return meaningful HTTP status codes

**Async/Await:**
- Use async for all I/O operations
- Proper exception handling in async context
- Use asyncio.gather for concurrent operations
- Avoid blocking calls in async functions

### 10.2 File Organization

**One concern per file:**
- Keep files under 300 lines
- Clear single responsibility
- Related functions grouped together

**Import organization:**
```python
# Standard library
import os
from typing import List

# Third-party
import numpy as np
from fastapi import FastAPI

# Local imports
from ..models import Document
from .chunker import Chunker
```

### 10.3 Testing Guidelines

**Test naming:**
```python
def test_<component>_<scenario>_<expected_behavior>():
    """Test that component does X when Y"""
    pass

# Examples:
def test_chunker_fixed_size_creates_equal_chunks()
def test_retriever_empty_query_raises_error()
def test_api_query_endpoint_returns_sources()
```

**Test structure (AAA pattern):**
```python
def test_example():
    # Arrange
    chunker = Chunker(max_tokens=100)
    document = Document(content="..." * 1000)
    
    # Act
    chunks = chunker.chunk_document(document)
    
    # Assert
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)
```

### 10.4 Logging Standards

**Use structured logging:**

```python
from loguru import logger

logger.info(
    "Document ingested",
    document_id=doc.id,
    doc_type=doc.doc_type,
    chunk_count=len(chunks),
    processing_time_ms=elapsed_ms
)

logger.error(
    "Vector store upsert failed",
    error=str(e),
    document_id=doc.id,
    retry_attempt=attempt
)
```

**Log levels:**
- DEBUG: Detailed diagnostic info
- INFO: Important events (ingestion, queries)
- WARNING: Unexpected but handled situations
- ERROR: Error events that still allow operation
- CRITICAL: Serious errors requiring immediate attention

### 10.5 Security Considerations

**API Keys:**
- Never commit API keys to git
- Use environment variables
- Validate API keys on startup
- Rotate keys regularly

**Input Validation:**
- Validate all user inputs with Pydantic
- Sanitize file paths (prevent directory traversal)
- Limit request sizes (max 10MB for documents)
- Rate limiting on API endpoints

**SQL Injection Prevention:**
- Use SQLAlchemy ORM (parameterized queries)
- Never construct SQL from user input
- Validate UUIDs before queries

### 10.6 Performance Optimization

**Database:**
- Create indexes on frequently queried columns
- Use connection pooling
- Batch inserts when possible

**Caching:**
- Cache embeddings with LRU cache
- Cache frequently accessed documents
- Invalidate cache on updates

**Async Operations:**
- Use asyncio.gather for parallel operations
- Limit concurrent operations (semaphore)
- Stream large responses

---

## 11. Success Criteria & Deliverables

### 11.1 Minimum Viable Product (MVP)

**Must Have:**
- ✅ Ingest PDFs and answer questions
- ✅ Vector search retrieval
- ✅ OpenAI integration for generation
- ✅ Basic REST API
- ✅ Docker deployment
- ✅ README with quick start

**Nice to Have:**
- Hybrid search
- Multiple LLM providers
- Streaming responses
- Evaluation framework

### 11.2 Production-Ready Version

**Must Have:**
- ✅ All document types (PDF, web, code, markdown)
- ✅ Hybrid retrieval + reranking
- ✅ Both OpenAI and Anthropic support
- ✅ Streaming WebSocket API
- ✅ Comprehensive test suite (>80% coverage)
- ✅ Automated evaluation with metrics
- ✅ CI/CD pipeline
- ✅ Complete documentation
- ✅ Docker Compose deployment

### 11.3 Final Deliverables Checklist

**Code:**
- [ ] All source code in src/
- [ ] Complete test suite in tests/
- [ ] Scripts for ingestion and evaluation
- [ ] Docker configuration

**Documentation:**
- [ ] README.md with quick start
- [ ] Architecture diagram
- [ ] API reference
- [ ] Deployment guide
- [ ] Evaluation results

**Infrastructure:**
- [ ] Dockerfile and docker-compose.yml
- [ ] GitHub Actions CI/CD
- [ ] Pre-commit hooks configured

**Evaluation:**
- [ ] Golden dataset with 50+ examples
- [ ] Automated test suite
- [ ] Performance benchmarks
- [ ] Cost analysis

**GitHub Repository:**
- [ ] Clean commit history
- [ ] MIT License
- [ ] .gitignore configured
- [ ] Issues and PR templates
- [ ] GitHub Actions badges in README

---

## 12. Common Pitfalls to Avoid

1. **Don't hardcode configuration**
   - Use environment variables and config classes

2. **Don't skip error handling**
   - Every API call can fail
   - Implement retry logic with exponential backoff

3. **Don't ignore token limits**
   - Always count tokens before LLM calls
   - Truncate context if needed

4. **Don't skip evaluation**
   - Metrics prove the system works
   - Golden dataset is essential

5. **Don't overcomplicate initially**
   - Start with simple implementations
   - Iterate and improve

6. **Don't forget logging**
   - Structured logging is essential for debugging
   - Log all important events

7. **Don't skip documentation**
   - README is the first thing recruiters see
   - Code without docs is incomplete

8. **Don't ignore costs**
   - Track token usage
   - Optimize for cost efficiency

---

## 13. Extension Ideas (Post-MVP)

Once the core system is working, consider these extensions:

1. **Feedback Loop**
   - Allow users to rate answers
   - Use feedback to improve retrieval

2. **Multi-Turn Conversations**
   - Track conversation history
   - Context-aware follow-up questions

3. **Advanced Chunking**
   - Sentence-window retrieval
   - Hierarchical chunking (document → section → paragraph)

4. **Query Classification**
   - Route different query types to different strategies
   - Factual vs analytical vs creative

5. **Document Updates**
   - Webhook for real-time updates
   - Incremental re-indexing

6. **Multi-Modal Support**
   - Image understanding with vision models
   - Table extraction and querying

7. **Fine-Tuned Models**
   - Fine-tune embedding model on domain data
   - Fine-tune reranker for better performance

8. **Advanced Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Alert on anomalies

---

## 14. Appendix

### 14.1 Recommended Reading

**RAG Fundamentals:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- LangChain documentation on RAG
- Pinecone RAG guide

**Evaluation:**
- RAGAS framework documentation
- "Evaluating RAG Systems" (various blog posts)

**Production ML:**
- "Designing Machine Learning Systems" by Chip Huyen
- "Building Machine Learning Powered Applications" by Emmanuel Ameisen

### 14.2 Useful Tools & Libraries

**Development:**
- black (code formatting)
- flake8 (linting)
- mypy (type checking)
- pytest (testing)
- httpx (async HTTP client)

**Monitoring:**
- loguru (logging)
- prometheus-client (metrics)
- opentelemetry (tracing)

**Documentation:**
- mkdocs (documentation site)
- mermaid (diagrams)
- swagger/openapi (API docs)

### 14.3 Sample Queries for Testing

**Factual Questions:**
- "Who invented the transformer architecture?"
- "What year was GPT-3 released?"

**Explanatory Questions:**
- "How does self-attention work in transformers?"
- "What is the difference between BERT and GPT?"

**Comparative Questions:**
- "Compare LSTM and transformer architectures"
- "What are the pros and cons of RAG vs fine-tuning?"

**Multi-Hop Questions:**
- "What company created the model that introduced attention mechanisms, and what year?"

**Ambiguous Questions:**
- "What is the best embedding model?" (subjective)
- "Tell me about Claude" (vague)

---

## Conclusion

This PRD provides a comprehensive specification for building a production-quality RAG system. The implementation should follow the phases outlined, maintaining focus on code quality, evaluation, and documentation throughout. The final deliverable will demonstrate deep understanding of RAG systems and serve as a strong portfolio piece for LLM engineering roles.

**Key Success Factors:**
1. Follow the architecture exactly
2. Implement comprehensive testing
3. Create golden dataset early
4. Document as you build
5. Optimize for both quality and cost
6. Deploy with Docker for easy reproducibility

**Final Note:** This system should take 4-5 weeks to build completely. The AI agent implementing this should follow phases sequentially, test thoroughly at each stage, and maintain high code quality throughout.
