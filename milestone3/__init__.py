"""
Milestone 3 — Meeting Knowledge Repository, Embeddings, Semantic Search & RAG

Adds semantic search and grounded question-answering on top of the
existing Milestone 2 meeting intelligence pipeline.

Modules:
    schemas              — Pydantic data contracts (SearchRequest, SearchResult, RAGRequest, RAGResponse)
    meeting_repository   — Read-only access to the Milestone 2 SQLite database
    chunking_service     — Deterministic overlap-aware transcript chunking for embedding
    embedding_service    — Gemini text-embedding API wrapper
    vector_store_service — ChromaDB vector store abstraction
    indexing_pipeline    — Orchestrates chunking → embedding → vector store upsert
    semantic_search_service — Natural-language query → ranked SearchResult list
    rag_service          — Grounded LLM question-answering over retrieved context
"""
