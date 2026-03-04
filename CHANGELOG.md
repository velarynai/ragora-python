# Changelog

All notable changes to the Ragora Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-04

### Breaking

- `chat` and `chat_stream` now use grouped request options:
  `generation`, `retrieval`, `agentic`, and `metadata`
- Removed flat chat parameters (`collection_id`, `products`, `model`,
  `temperature`, `top_k`, `mode`, `system_prompt`, `session_id`, etc.)
  from top-level chat method signatures

### Added

- Restored agent API surface and models referenced by docs/examples:
  `create_agent`, `list_agents`, `get_agent`, `update_agent`, `delete_agent`,
  `agent_chat`, `agent_chat_stream`, `list_agent_sessions`,
  `get_agent_session`, `delete_agent_session`
- OpenAI-style chat completion retrieval parity with support for:
  `version_mode`, `document_keys`, `domain`, `domain_filter_mode`,
  `graph_filter`, and `temporal_filter`
- Agent auto-retrieval policy support on agent create/update via `retrieval_policy`
- `agent_chat` and `agent_chat_stream` now use auto retrieval only (message/session inputs, with optional `collection_ids` scope)
- Added optional `collection_ids` on `agent_chat` / `agent_chat_stream` for session-level collection scoping
- Added name-based reference resolution for SDK inputs:
  `collection` (UUID/slug/name) and `products` (UUID/slug/title)
- Kept backward compatibility with legacy `collection_id` / `product_ids`
  parameters and explicit conflict errors when both forms are passed
- `ThinkingStep` events in streaming for real-time agent status updates
- Exported typed error subclasses: `AuthenticationError`, `AuthorizationError`,
  `NotFoundError`, `RateLimitError`, `ServerError`
- Exported chat sub-option types: `ChatGenerationOptions`,
  `ChatRetrievalOptions`, `ChatAgenticOptions`, `ChatMetadataOptions`
- Exported `RagoraCitation` type for structured citation access
- Automatic retry with exponential backoff for 429 and 5xx errors

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
