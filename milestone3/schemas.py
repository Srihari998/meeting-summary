"""
Milestone 3 Pydantic Schemas
Defines data contracts for semantic search, RAG, and API endpoints.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Input model for a semantic search query."""

    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of results to return")
    source_type: Optional[Literal["transcript", "summary", "decision", "action_item"]] = Field(
        None,
        description="Optional filter: restrict search to a specific content type",
    )
    meeting_id: Optional[str] = Field(
        None,
        description="Optional filter: restrict search to a specific meeting ID",
    )
    date_from: Optional[str] = Field(
        None,
        description="Optional filter: ISO date string for start date",
    )
    date_to: Optional[str] = Field(
        None,
        description="Optional filter: ISO date string for end date",
    )


class SearchResult(BaseModel):
    """A single semantic search result, traceable to a stored meeting."""

    meeting_id: str = Field(..., description="Identifier of the source meeting")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score (0–1)")
    source_type: str = Field(..., description="Content type: transcript | summary | decision | action_item")
    content: str = Field(..., description="The retrieved text snippet")
    search_latency_ms: float = Field(0.0, ge=0.0, description="End-to-end search latency in milliseconds")


class SearchResponse(BaseModel):
    """Response model for the /search endpoint."""

    query: str = Field(..., description="The original search query")
    results: List[SearchResult] = Field(default_factory=list, description="Ranked list of search results")
    total_results: int = Field(0, ge=0, description="Number of results returned")
    search_latency_ms: float = Field(0.0, ge=0.0, description="Search latency in milliseconds")


class RAGRequest(BaseModel):
    """Input model for a RAG (retrieval-augmented generation) question."""

    question: str = Field(..., min_length=1, description="Natural-language question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve for grounding")


class RAGResponse(BaseModel):
    """Grounded LLM answer with full source traceability."""

    answer: str = Field(..., description="LLM-generated answer grounded in retrieved meeting context")
    meeting_ids: List[str] = Field(
        default_factory=list,
        description="Unique meeting IDs that contributed context to this answer",
    )
    sources: List[SearchResult] = Field(
        default_factory=list,
        description="The retrieved chunks that were used as context",
    )
    latency_ms: float = Field(0.0, ge=0.0, description="Total RAG pipeline latency in milliseconds")


class MeetingSummaryResponse(BaseModel):
    """Summary representation of a meeting in the /meetings listing."""

    id: str
    created_at: str
    summary: str
    key_points: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    action_items_count: int = 0
    participants_count: int = 0


class MeetingListResponse(BaseModel):
    """Response model for GET /meetings."""

    meetings: List[MeetingSummaryResponse] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class MeetingDetailResponse(BaseModel):
    """Detailed representation of a single meeting in GET /meetings/{id}."""

    id: str
    transcript: str
    summary: str
    key_points: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    participants: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str
