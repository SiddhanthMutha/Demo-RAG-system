# Frontend + Evaluation Framework Implementation Plan

## Overview
Add server-rendered Jinja2 frontend with HTMX interactivity and a full evaluation framework to the existing RAG system.

## Decisions

### Frontend
- **Templates**: Jinja2 via `Jinja2Templates` in FastAPI
- **Interactivity**: HTMX for async interactions
- **Styling**: Pico CSS via CDN (no build step)
- **Streaming**: SSE endpoint (`/api/v1/query/stream/sse`) for live token display
- **Pages**: Query, Ingestion, Documents list, Evaluation dashboard

### Evaluation
- **Scope**: Full suite (retrieval + generation + end-to-end)
- **Dataset**: Synthetic Q&A auto-generated from ingested documents
- **Storage**: Existing `EvaluationResult` table

## Tasks

### Phase 1: Frontend Foundation
1. Install `jinja2` (if not present), configure `Jinja2Templates` in `main.py`
2. Create directory `src/templates/` with base layout
3. Create `src/api/routes/views.py` with HTML-rendering routes
4. Add `/web/query`, `/web/ingest`, `/web/documents`, `/web/eval` routes
5. Create SSE streaming endpoint for query page
6. Add static file serving for CSS/JS if needed

### Phase 2: Evaluation Framework - Core
1. Create `src/evaluation/metrics.py`:
   - `precision_at_k(results, expected_ids, k)`
   - `recall_at_k(results, expected_ids, k)`
   - `mrr(results, expected_ids)`
   - `ndcg(results, expected_ids)`
2. Create `src/evaluation/judge.py`:
   - `check_faithfulness(context, answer)` using `build_faithfulness_check_prompt`
   - `check_answer_correctness(expected, generated)`
3. Create `src/evaluation/harness.py`:
   - `run_evaluation(queries, top_k)` -> metrics dict
   - Reuse existing `_run_retrieval` logic
4. Create `src/evaluation/dataset.py`:
   - `generate_synthetic_qa(document_chunks)` -> list of Q/A pairs

### Phase 3: Evaluation API
1. Create `src/api/routes/eval.py`:
   - `POST /api/v1/eval/run` -> trigger evaluation
   - `GET /api/v1/eval/results` -> list historical results
   - `GET /api/v1/eval/metrics` -> aggregated metrics
2. Add `EvaluationRepository` usage for persistence
3. Wire router in `main.py`

### Phase 4: CLI
1. Create `src/evaluation/cli.py`:
   - `run` command: execute evaluation, output JSON/table
   - `--synthetic` flag for auto-generated dataset

### Phase 5: Testing
1. Unit tests for metrics functions
2. Integration tests for eval endpoints
3. Test frontend pages render correctly

## Risks
- SSE endpoint race conditions under load
- Evaluation LLM costs if many queries run
- Template directory may need `.env` reloading in dev

## Open Questions
- None at this time