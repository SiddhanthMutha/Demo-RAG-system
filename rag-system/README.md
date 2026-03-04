# RAG System - Intelligent Document Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that lets you chat with your documents. Built with FastAPI, OpenAI, and Pinecone.

## What This Does

Ever wanted to ask questions about a pile of PDFs or code files without reading through them all? This system does exactly that:

- **Upload documents** (PDFs, code files, web pages, markdown)
- **Ask questions** in natural language
- **Get accurate answers** based on the content of your documents

The system uses a hybrid search approach - combining semantic search with keyword matching - to find the most relevant information before generating answers with GPT-4 or Claude.

## The Stack

| Component | Technology |
|-----------|------------|
| **API Framework** | FastAPI with async support |
| **Vector Database** | Pinecone for semantic search |
| **LLM Providers** | OpenAI (GPT-4) & Anthropic (Claude) |
| **Embeddings** | OpenAI Ada-002 or MiniLM (local) |
| **Metadata Storage** | PostgreSQL with pgvector |
| **Reranking** | Cross-encoder models |
| **Keyword Search** | BM25 algorithm |

## Architecture Highlights

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Documents │────▶│  Chunking   │────▶│  Vector Store   │
│   (PDF/Code)│     │  & Embedding│     │   (Pinecone)    │
└─────────────┘     └─────────────┘     └─────────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Answer    │◀────│  LLM + RAG  │◀────│  Hybrid Search  │
│   (Stream)  │     │  (GPT-4)    │     │  (Dense+Sparse) │
└─────────────┘     └─────────────┘     └─────────────────┘
                                                ▲
                                         ┌─────────────┐
                                         │   Query     │
                                         └─────────────┘
```

## Key Features

- **Multiple Document Types**: PDFs, Python code, Markdown, web pages
- **Hybrid Retrieval**: Combines vector similarity with BM25 keyword search
- **Smart Reranking**: Cross-encoder reorders results for better relevance
- **Streaming Responses**: Real-time token streaming via WebSocket
- **Document Chunking**: Semantic and recursive chunking strategies
- **Query Rewriting**: Automatic query expansion for better retrieval
- **Full Evaluation Suite**: Golden dataset testing, performance benchmarks

## Quick Start

### 1. Clone and Setup

```bash
cd rag-system
cp .env.example .env
# Edit .env with your API keys
```

### 2. Required API Keys

You'll need:
- **OpenAI API Key** - for embeddings and LLM generation
- **Pinecone API Key** - for vector storage
- **(Optional) Anthropic Key** - for Claude as an alternative LLM

### 3. Run with Docker

```bash
cd infra/docker
docker-compose up -d
```

### 4. Start the API

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

## Using the API

### Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
  -F "file=@document.pdf" \
  -F "chunking_strategy=recursive"
```

### Ask a Question

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main conclusions?",
    "top_k": 5,
    "stream": false
  }'
```

### Stream a Response

```bash
# Via WebSocket
websocat ws://localhost:8000/api/v1/query/stream
```

## Project Structure

```
rag-system/
├── src/
│   ├── api/              # FastAPI routes & middleware
│   ├── ingestion/        # Document parsers & chunkers
│   ├── retrieval/        # Vector search, BM25, reranker
│   ├── generation/       # LLM clients & prompt building
│   └── database/         # SQLAlchemy models & repository
├── tests/                # Unit, integration & eval tests
├── scripts/              # Helper scripts
├── infra/                # Docker & Kubernetes configs
└── docs/                 # Architecture & API docs
```

## Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Evaluation with golden dataset
python scripts/run_evaluation.py
```

## What I Built

This project implements a complete RAG pipeline from scratch:

1. **Document Ingestion Pipeline** - Parses PDFs, code files, and web pages with intelligent chunking
2. **Hybrid Search System** - Combines dense vector search (Pinecone) with sparse keyword search (BM25)
3. **Cross-Encoder Reranking** - Reorders retrieved chunks for maximum relevance
4. **Multi-Provider LLM Client** - Supports OpenAI and Anthropic with streaming
5. **Production API** - FastAPI with middleware for logging, error handling, and rate limiting
6. **Evaluation Framework** - Performance benchmarks and quality metrics
7. **Docker Deployment** - Complete containerization with docker-compose

## Why These Choices?

- **FastAPI**: Async support + automatic API docs
- **Pinecone**: Managed vector DB with excellent performance
- **Hybrid Search**: Pure vector search misses exact keyword matches; hybrid catches both
- **BM25**: Still unbeatable for keyword relevance
- **Cross-Encoder**: More accurate than dot-product similarity for reranking

## Roadmap

- [ ] Web interface (React frontend)
- [ ] Conversation memory
- [ ] Multi-modal support (images)
- [ ] Fine-tuned embedding models
- [ ] GraphRAG integration

---

Built with Python, lots of coffee, and a healthy obsession with retrieval accuracy.
