# Changelog

All notable changes to the Ragora Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-02-19

### Added

- Agent CRUD operations (`create_agent`, `get_agent`, `list_agents`, `update_agent`, `delete_agent`)
- Agent chat with streaming support (`agent_chat`, `agent_chat_stream`)
- Agent session management (`list_agent_sessions`, `get_agent_session`, `delete_agent_session`)
- Pydantic models for agents, sessions, and messages
- Document upload metadata fields (`custom_tags`, `domain`, `source_type`, `release_tag`, `version`, `effective_at`, `expires_at`, etc.)

### Changed

- Agentic RAG example rewritten to use server-side agent chat instead of local LangGraph orchestration
- Removed LangGraph, LangChain, and multi-provider LLM dependencies from agentic RAG example

### Fixed

- Streaming timeout now resets on each chunk instead of using a fixed wall-clock timeout

## [0.1.0] - 2026-02-07

### Added

- Initial release
- Async-first client with `httpx`
- Full type hints with Pydantic models
- Collections CRUD operations
- Document upload with processing status tracking
- Hybrid search (dense + sparse vectors)
- Chat completions with streaming support
- Marketplace browsing
- Credits and balance management
- Response metadata extraction (cost, rate limits, request IDs)
- Error handling with `RagoraException`
- Agentic RAG module with LangGraph integration
- Multi-provider LLM support (OpenAI, Anthropic, Azure, Google, Ollama)
