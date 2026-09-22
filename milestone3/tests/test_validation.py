"""
Validation Test Suite — Milestone 3 (Task 7)
Validates semantic search retrieval accuracy, multi-meeting relevance,
metadata/date filtering, traceability, context construction, and grounded RAG answers.
"""

from datetime import datetime, timezone
import pytest
from unittest.mock import MagicMock, patch

from milestone3.schemas import SearchRequest, SearchResult, RAGRequest, RAGResponse
from milestone3.semantic_search_service import SemanticSearchService
from milestone3.rag_service import RAGService
from milestone3.vector_store_service import VectorStoreService
from milestone3.embedding_service import EmbeddingService
from milestone2.models import Meeting, ActionItemRecord, ParticipantRecord


# ------------------------------------------------------------------------------
# Helpers & Fixtures
# ------------------------------------------------------------------------------

def _cosine_sim(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class MockEmbeddingService(EmbeddingService):
    """
    Deterministic embedding service for testing:
    Maps semantic concepts to distinct vector directions so cosine similarity behaves realistically.
    """
    def __init__(self):
        self._model = "test-mock-model"
        self._api_keys = ["mock-key"]

    def embed_text(self, text: str):
        t = text.lower()
        # 4-dimensional vector: [database/migration, mobile/ui, mars/space, timeline/deadline]
        vec = [0.05, 0.05, 0.05, 0.05]
        if any(w in t for w in ["database", "postgres", "mysql", "migration", "schema"]):
            vec[0] += 1.0
        if any(w in t for w in ["mobile", "ui", "testing", "app", "interface", "priya"]):
            vec[1] += 1.0
        if any(w in t for w in ["mars", "space", "rocket", "orbit"]):
            vec[2] += 1.0
        if any(w in t for w in ["friday", "deadline", "timeline", "schedule", "date"]):
            vec[3] += 1.0

        # Normalize
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_batch(self, texts, skip_hashes=None):
        return [self.embed_text(t) if t else None for t in texts]


@pytest.fixture
def memory_vector_store(tmp_path):
    """Creates a fresh isolated ChromaDB vector store."""
    return VectorStoreService(persist_directory=str(tmp_path / "chroma_test"))


@pytest.fixture
def mock_embedder():
    return MockEmbeddingService()


# ------------------------------------------------------------------------------
# 7.1 & 7.2 & 7.3: Search Retrieval Accuracy & Edge Queries
# ------------------------------------------------------------------------------

class TestSemanticSearchValidation:
    def test_7_1_relevant_meeting_retrieval(self, memory_vector_store, mock_embedder):
        """Meeting A (database migration) vs Meeting B (mobile UI). Search for migration returns Meeting A."""
        # Index Meeting A
        text_a = "The team discussed migrating the database from MySQL to PostgreSQL."
        memory_vector_store.upsert(
            chunk_id="mtg_a_1",
            text=text_a,
            embedding=mock_embedder.embed_text(text_a),
            meeting_id="mtg_A",
            source_type="transcript",
            content_hash="hash_a",
            chunk_index=0,
        )

        # Index Meeting B
        text_b = "The team discussed mobile application UI testing and user feedback."
        memory_vector_store.upsert(
            chunk_id="mtg_b_1",
            text=text_b,
            embedding=mock_embedder.embed_text(text_b),
            meeting_id="mtg_B",
            source_type="transcript",
            content_hash="hash_b",
            chunk_index=0,
        )

        search_svc = SemanticSearchService(embedding_svc=mock_embedder, vector_store=memory_vector_store)
        results = search_svc.search(SearchRequest(query="Which meeting discussed database migration?", top_k=2))

        assert len(results) >= 1
        top_result = results[0]
        assert top_result.meeting_id == "mtg_A"
        assert "migrating the database" in top_result.content
        assert top_result.relevance_score > 0.7

    def test_7_2_irrelevant_query_behavior(self, memory_vector_store, mock_embedder):
        """Query for Mars mission when only software meetings exist -> very low relevance or distinct separation."""
        text_a = "Database indexing and PostgreSQL migration plan."
        memory_vector_store.upsert(
            chunk_id="mtg_a_1",
            text=text_a,
            embedding=mock_embedder.embed_text(text_a),
            meeting_id="mtg_A",
            source_type="transcript",
            content_hash="hash_a",
            chunk_index=0,
        )

        search_svc = SemanticSearchService(embedding_svc=mock_embedder, vector_store=memory_vector_store)
        results = search_svc.search(SearchRequest(query="What was discussed about a Mars rocket mission?", top_k=1))

        assert len(results) == 1
        # Relevance score should reflect the orthogonal semantic vector
        assert results[0].relevance_score < 0.5

    def test_7_3_multiple_matching_meetings(self, memory_vector_store, mock_embedder):
        """Meetings 1 & 2 discuss database migration, Meeting 3 discusses mobile app. Both 1 & 2 retrieved."""
        m1 = "Database migration was discussed and scheduled."
        m2 = "PostgreSQL migration timeline was reviewed by backend engineers."
        m3 = "Mobile application testing was discussed."

        for mid, text in [("mtg_1", m1), ("mtg_2", m2), ("mtg_3", m3)]:
            memory_vector_store.upsert(
                chunk_id=f"{mid}_c0",
                text=text,
                embedding=mock_embedder.embed_text(text),
                meeting_id=mid,
                source_type="transcript",
                content_hash=f"h_{mid}",
                chunk_index=0,
            )

        search_svc = SemanticSearchService(embedding_svc=mock_embedder, vector_store=memory_vector_store)
        results = search_svc.search(SearchRequest(query="Tell me about database migration discussions.", top_k=2))

        retrieved_ids = [r.meeting_id for r in results]
        assert "mtg_1" in retrieved_ids
        assert "mtg_2" in retrieved_ids
        assert "mtg_3" not in retrieved_ids

    def test_7_9_empty_search_results_on_empty_store(self, memory_vector_store, mock_embedder):
        """Empty vector store returns clean empty list, not error."""
        search_svc = SemanticSearchService(embedding_svc=mock_embedder, vector_store=memory_vector_store)
        results = search_svc.search(SearchRequest(query="Any meeting query"))
        assert results == []


# ------------------------------------------------------------------------------
# 7.4 & 7.5: Metadata & Date Filtering Validation
# ------------------------------------------------------------------------------

class TestMetadataAndDateFiltering:
    def test_7_4_date_range_filtering(self, memory_vector_store, mock_embedder):
        """Meetings filtered by date range exclude out-of-range records."""
        t1 = "Database migration review meeting"
        t2 = "Database optimization discussion"

        memory_vector_store.upsert(
            chunk_id="mtg_old_c0",
            text=t1,
            embedding=mock_embedder.embed_text(t1),
            meeting_id="mtg_old",
            source_type="transcript",
            content_hash="h1",
            chunk_index=0,
        )
        memory_vector_store.upsert(
            chunk_id="mtg_new_c0",
            text=t2,
            embedding=mock_embedder.embed_text(t2),
            meeting_id="mtg_new",
            source_type="transcript",
            content_hash="h2",
            chunk_index=0,
        )

        m_old = MagicMock(spec=Meeting)
        m_old.id = "mtg_old"
        m_old.created_at = datetime(2026, 8, 15, tzinfo=timezone.utc)

        m_new = MagicMock(spec=Meeting)
        m_new.id = "mtg_new"
        m_new.created_at = datetime(2026, 9, 10, tzinfo=timezone.utc)

        with patch("milestone3.meeting_repository.MeetingRepository.get_meeting_by_id") as mock_get:
            mock_get.side_effect = lambda mid: m_old if mid == "mtg_old" else m_new

            search_svc = SemanticSearchService(embedding_svc=mock_embedder, vector_store=memory_vector_store)
            
            # Filter for September only
            results = search_svc.search(
                SearchRequest(
                    query="database",
                    date_from="2026-09-01T00:00:00",
                    date_to="2026-09-30T23:59:59",
                )
            )

            meeting_ids = [r.meeting_id for r in results]
            assert "mtg_new" in meeting_ids
            assert "mtg_old" not in meeting_ids

    def test_7_5_source_type_and_meeting_id_filtering(self, memory_vector_store, mock_embedder):
        """Filters by source_type ('decision' vs 'transcript') and specific meeting_id."""
        memory_vector_store.upsert(
            chunk_id="m1_transcript",
            text="Team discussed database migration options.",
            embedding=mock_embedder.embed_text("Team discussed database migration options."),
            meeting_id="m1",
            source_type="transcript",
            content_hash="h_tr",
            chunk_index=0,
        )
        memory_vector_store.upsert(
            chunk_id="m1_decision",
            text="Decided to proceed with PostgreSQL database migration.",
            embedding=mock_embedder.embed_text("Decided to proceed with PostgreSQL database migration."),
            meeting_id="m1",
            source_type="decision",
            content_hash="h_dec",
            chunk_index=0,
        )

        search_svc = SemanticSearchService(embedding_svc=mock_embedder, vector_store=memory_vector_store)

        # 1. Filter by source_type = 'decision'
        dec_results = search_svc.search(
            SearchRequest(query="database migration", source_type="decision")
        )
        assert len(dec_results) == 1
        assert dec_results[0].source_type == "decision"

        # 2. Filter by meeting_id = 'm1'
        m1_results = search_svc.search(
            SearchRequest(query="database migration", meeting_id="m1")
        )
        assert all(r.meeting_id == "m1" for r in m1_results)


# ------------------------------------------------------------------------------
# 7.6, 7.7, 7.8: Traceability, Context Retrieval & Grounded RAG Answers
# ------------------------------------------------------------------------------

class TestRAGTraceabilityAndGrounding:
    def test_7_6_correct_source_meeting_mapping(self, memory_vector_store, mock_embedder):
        """Verifies vector -> meeting_id -> relational meeting mapping."""
        content = "Alice: We approved PostgreSQL migration."
        memory_vector_store.upsert(
            chunk_id="mtg_100_summary",
            text=content,
            embedding=mock_embedder.embed_text(content),
            meeting_id="mtg_100",
            source_type="summary",
            content_hash="h100",
            chunk_index=0,
        )

        search_svc = SemanticSearchService(embedding_svc=mock_embedder, vector_store=memory_vector_store)
        results = search_svc.search(SearchRequest(query="PostgreSQL migration"))

        assert len(results) == 1
        assert results[0].meeting_id == "mtg_100"
        assert results[0].content == content

    def test_7_7_correct_context_retrieval_for_rag(self):
        """Verifies RAG prompt context builder includes excerpts with traceability tags."""
        svc = RAGService.__new__(RAGService)
        sources = [
            SearchResult(
                meeting_id="mtg-alpha",
                relevance_score=0.95,
                source_type="action_item",
                content="Priya will complete mobile application testing by Friday.",
                search_latency_ms=25.0,
            )
        ]
        context = svc._build_context(sources)
        assert "[Excerpt 1 | Meeting: mtg-alpha | Type: action_item]" in context
        assert "Priya will complete mobile application testing by Friday." in context

    def test_7_8_grounded_ai_answer(self, memory_vector_store, mock_embedder):
        """
        Meeting context contains deadline 'Friday'.
        RAGService passes retrieved context to LLM and returns grounded answer with meeting ID citation.
        """
        meeting_content = "Action item: Priya will complete the mobile application testing by Friday."
        memory_vector_store.upsert(
            chunk_id="mtg_priya_ai",
            text=meeting_content,
            embedding=mock_embedder.embed_text(meeting_content),
            meeting_id="mtg_priya",
            source_type="action_item",
            content_hash="hpriya",
            chunk_index=0,
        )

        search_svc = SemanticSearchService(embedding_svc=mock_embedder, vector_store=memory_vector_store)

        rag_svc = RAGService.__new__(RAGService)
        rag_svc._search_svc = search_svc
        rag_svc._model = "test-model"
        rag_svc._prompt_template = "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"

        # Mock LLM client returning grounded answer
        mock_response = MagicMock()
        mock_response.text = "The deadline decided for the mobile application testing is Friday, assigned to Priya [Meeting: mtg_priya]."
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        rag_svc._client = mock_client

        response = rag_svc.answer(RAGRequest(question="What deadline was decided for the mobile application testing?"))

        assert isinstance(response, RAGResponse)
        assert "Friday" in response.answer
        assert "mtg_priya" in response.meeting_ids
        assert len(response.sources) == 1
        assert "Priya will complete the mobile application testing by Friday" in response.sources[0].content

        # Verify LLM was actually called with the retrieved context
        call_kwargs = mock_client.models.generate_content.call_args[1]
        sent_prompt = call_kwargs["contents"]
        assert "Priya will complete the mobile application testing by Friday" in sent_prompt
        assert "What deadline was decided for the mobile application testing?" in sent_prompt
