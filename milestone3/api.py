"""
FastAPI REST API Layer — Milestone 3 (Task 6)
Exposes endpoints for historical meetings, semantic search, and RAG QA.

Endpoints:
  GET  /meetings         - Retrieve historical meetings list (with pagination)
  GET  /meetings/{id}    - Retrieve single meeting details with summary, transcript, decisions, action items, participants
  POST /search           - Semantic search across meeting knowledge repository
  POST /ask              - Grounded RAG question answering with citations
  GET  /health           - System health probe
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Header, Query, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

# Configure logger
logger = logging.getLogger("intellimeet.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

try:
    from milestone3.schemas import (
        MeetingDetailResponse,
        MeetingListResponse,
        MeetingSummaryResponse,
        RAGRequest,
        RAGResponse,
        SearchRequest,
        SearchResponse,
        SearchResult,
    )
    from milestone3.meeting_repository import MeetingRepository
    from milestone3.semantic_search_service import SemanticSearchService
    from milestone3.rag_service import RAGService, RAGError
    from milestone3.embedding_service import EmbeddingError
    from milestone3.vector_store_service import VectorStoreError, VectorStoreService
    from milestone3.indexing_pipeline import index_meeting, index_all_meetings
except ImportError:
    from schemas import (  # type: ignore
        MeetingDetailResponse,
        MeetingListResponse,
        MeetingSummaryResponse,
        RAGRequest,
        RAGResponse,
        SearchRequest,
        SearchResponse,
        SearchResult,
    )
    from meeting_repository import MeetingRepository  # type: ignore
    from semantic_search_service import SemanticSearchService  # type: ignore
    from rag_service import RAGService, RAGError  # type: ignore
    from embedding_service import EmbeddingError  # type: ignore
    from vector_store_service import VectorStoreError, VectorStoreService  # type: ignore
    from indexing_pipeline import index_meeting, index_all_meetings  # type: ignore

app = FastAPI(
    title="IntelliMeet API",
    description="Intelligent Meeting Intelligence & Semantic Knowledge Repository API",
    version="3.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Authentication Dependency ──────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


def verify_authentication(
    x_api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth),
) -> bool:
    """
    Validates API requests using X-API-Key or Bearer token against configured credentials.
    If API_AUTH_TOKEN is configured in environment, incoming requests MUST match.
    If not configured in environment, authentication is open (development/internal mode).
    """
    expected_token = os.environ.get("API_AUTH_TOKEN") or os.environ.get("API_KEY")
    if not expected_token:
        # Auth disabled or open in dev mode
        return True

    provided_token = None
    if x_api_key:
        provided_token = x_api_key.strip()
    elif bearer and bearer.credentials:
        provided_token = bearer.credentials.strip()

    if not provided_token or provided_token != expected_token.strip():
        logger.warning("[Auth] Unauthorized request attempt with token: %s", "provided" if provided_token else "missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


# ── Middleware for Request Logging & Timing ──────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    logger.info("[HTTP] %s %s from %s", request.method, request.url.path, client_host)

    try:
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        logger.info("[HTTP] %s %s -> %d (%.2f ms)", request.method, request.url.path, response.status_code, process_time_ms)
        return response
    except Exception as exc:
        process_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error("[HTTP] %s %s failed after %.2f ms: %s", request.method, request.url.path, process_time_ms, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred.", "error_type": type(exc).__name__},
        )


# ── Global Exception Handlers ────────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning("[API] ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(RAGError)
async def rag_error_handler(request: Request, exc: RAGError):
    logger.error("[API] RAGError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"RAG service error: {exc}"},
    )


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request: Request, exc: EmbeddingError):
    logger.error("[API] EmbeddingError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Embedding generation failed: {exc}"},
    )


@app.exception_handler(VectorStoreError)
async def vector_store_error_handler(request: Request, exc: VectorStoreError):
    logger.error("[API] VectorStoreError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Vector store operation failed: {exc}"},
    )


# ── Dependency Injection Providers ───────────────────────────────────────────
def get_repository() -> MeetingRepository:
    return MeetingRepository()


def get_search_service() -> SemanticSearchService:
    return SemanticSearchService()


def get_rag_service() -> RAGService:
    return RAGService()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check(
    repo: MeetingRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """System health check and dependency status."""
    try:
        meeting_count = repo.count_meetings()
        db_status = "connected"
    except Exception as exc:
        meeting_count = 0
        db_status = f"error: {exc}"

    try:
        vstore = VectorStoreService()
        vector_count = vstore.count()
        vstore_status = "connected"
    except Exception as exc:
        vector_count = 0
        vstore_status = f"error: {exc}"

    return {
        "status": "healthy" if db_status == "connected" and vstore_status == "connected" else "degraded",
        "database": db_status,
        "meetings_in_db": meeting_count,
        "vector_store": vstore_status,
        "chunks_indexed": vector_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/meetings",
    response_model=MeetingListResponse,
    tags=["Meetings"],
    summary="List historical meetings",
)
def list_meetings(
    limit: int = Query(50, ge=1, le=500, description="Number of records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    authenticated: bool = Depends(verify_authentication),
    repo: MeetingRepository = Depends(get_repository),
) -> MeetingListResponse:
    """
    Returns historical meeting records with pagination and summary intelligence.
    Eagerly loads action items and participants.
    """
    all_meetings = repo.get_all_meetings()
    total = len(all_meetings)

    paginated = all_meetings[offset: offset + limit]

    meeting_items = []
    for m in paginated:
        meeting_items.append(
            MeetingSummaryResponse(
                id=m.id,
                created_at=m.created_at.isoformat() if m.created_at else "",
                summary=m.summary or "",
                key_points=m.key_points or [],
                decisions=m.decisions or [],
                action_items_count=len(m.action_items) if m.action_items else 0,
                participants_count=len(m.participants) if m.participants else 0,
            )
        )

    return MeetingListResponse(
        meetings=meeting_items,
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/meetings/{meeting_id}",
    response_model=MeetingDetailResponse,
    tags=["Meetings"],
    summary="Get single meeting details",
)
def get_meeting(
    meeting_id: str,
    authenticated: bool = Depends(verify_authentication),
    repo: MeetingRepository = Depends(get_repository),
) -> MeetingDetailResponse:
    """
    Retrieves complete meeting record by ID including transcript, summary,
    decisions, action items, participants, and creation timestamp.
    """
    meeting = repo.get_meeting_by_id(meeting_id)
    if meeting is None:
        logger.warning("[API] Meeting not found: %s", meeting_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting '{meeting_id}' not found.",
        )

    action_items_data = []
    if meeting.action_items:
        for ai in meeting.action_items:
            action_items_data.append(
                {
                    "id": ai.id,
                    "task": ai.task,
                    "assignee": ai.assignee,
                    "deadline": ai.deadline,
                    "priority": ai.priority,
                    "status": ai.status,
                }
            )

    participants_data = []
    if meeting.participants:
        for p in meeting.participants:
            participants_data.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "is_unknown_assignee": p.is_unknown_assignee,
                }
            )

    return MeetingDetailResponse(
        id=meeting.id,
        transcript=meeting.transcript or "",
        summary=meeting.summary or "",
        key_points=meeting.key_points or [],
        decisions=meeting.decisions or [],
        action_items=action_items_data,
        participants=participants_data,
        created_at=meeting.created_at.isoformat() if meeting.created_at else "",
    )


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Semantic search over meeting knowledge repository",
)
def search_meetings(
    request: SearchRequest,
    authenticated: bool = Depends(verify_authentication),
    search_svc: SemanticSearchService = Depends(get_search_service),
) -> SearchResponse:
    """
    Executes semantic search against ChromaDB using dense Gemini embeddings.
    Returns ranked search results traceable to source meetings.
    """
    logger.info("[Search] Query: '%s' (top_k=%d, source_type=%s)", request.query, request.top_k, request.source_type)
    results: List[SearchResult] = search_svc.search(request)

    latency = results[0].search_latency_ms if results else 0.0
    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results),
        search_latency_ms=latency,
    )


@app.post(
    "/ask",
    response_model=RAGResponse,
    tags=["RAG"],
    summary="Ask questions grounded in meeting history",
)
def ask_rag(
    request: RAGRequest,
    authenticated: bool = Depends(verify_authentication),
    rag_svc: RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """
    Retrieval-Augmented Generation (RAG) question answering.
    Retrieves relevant meeting excerpts and generates an accurate, grounded answer citing source meetings.
    """
    logger.info("[RAG] Question: '%s' (top_k=%d)", request.question, request.top_k)
    response: RAGResponse = rag_svc.answer(request)
    return response


@app.post(
    "/index",
    tags=["Indexing"],
    summary="Trigger indexing of meetings into vector repository",
)
def trigger_indexing(
    meeting_id: Optional[str] = Query(None, description="Optional single meeting ID to index. If omitted, indexes all."),
    authenticated: bool = Depends(verify_authentication),
) -> Dict[str, Any]:
    """Indexes meeting(s) from SQLite DB into ChromaDB."""
    if meeting_id:
        res = index_meeting(meeting_id)
    else:
        res = index_all_meetings()

    return {
        "success": res.success,
        "chunks_indexed": res.chunks_indexed,
        "chunks_skipped": res.chunks_skipped,
        "meetings_processed": res.meeting_ids_processed,
        "errors": res.errors,
    }
