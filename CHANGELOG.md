# Changelog

All notable changes to the Ragora Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
