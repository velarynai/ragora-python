"""
Agentic RAG Agent

A concise LangGraph-based agent that:
1. Plans search queries from the user question
2. Retrieves from Ragora (optionally over multiple iterations)
3. Evaluates context sufficiency
4. Synthesizes a final cited answer
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import fields
from datetime import datetime
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:
    AsyncSqliteSaver = None

from ragora import RagoraClient

from .config import AgentConfig, LLMConfig, LLMProvider
from .state import AgentPhase, AgentState, create_initial_state
from .tools import (
    ANSWER_SYNTHESIS_PROMPT,
    CONTEXT_EVALUATION_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    ToolFactory,
    format_context_for_prompt,
    format_search_history,
    parse_json_response,
)


DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-5.3-codex",
    LLMProvider.ANTHROPIC: "claude-opus-4.6",
    LLMProvider.GOOGLE: "gemini-3-pro",
    LLMProvider.OLLAMA: "llama-4-70b",
    LLMProvider.OPENROUTER: "google/gemini-3-flash-preview",
}


def _extract_text(content: Any) -> str:
    """Normalize LangChain response content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if maybe_text:
                    parts.append(str(maybe_text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def _normalize_queries(
    candidate_queries: Any,
    fallback_query: str,
    allow_decomposition: bool,
    max_queries: int,
) -> list[str]:
    if not allow_decomposition:
        return [fallback_query]

    if isinstance(candidate_queries, str):
        raw_queries = [candidate_queries]
    elif isinstance(candidate_queries, list):
        raw_queries = [str(q) for q in candidate_queries if isinstance(q, (str, int, float))]
    else:
        raw_queries = []

    cleaned: list[str] = []
    seen: set[str] = set()
    for query in raw_queries:
        normalized = query.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
        if len(cleaned) >= max_queries:
            break

    return cleaned or [fallback_query]


def _raise_llm_error_context(llm: BaseChatModel, exc: Exception) -> None:
    llm_model = getattr(llm, "model_name", getattr(llm, "model", "<unknown>"))
    llm_base = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", None)
    details = f"LLM call failed (model={llm_model}, base_url={llm_base}): {exc}"
    raise RuntimeError(details) from exc


def create_llm(config: AgentConfig) -> BaseChatModel:
    """Create a LangChain chat model based on configuration."""
    llm_config = config.llm
    kwargs = llm_config.get_model_kwargs()

    if llm_config.provider == LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("Install `langchain-openai` to use OPENAI provider.") from exc
        return ChatOpenAI(model=llm_config.model, **kwargs)

    if llm_config.provider == LLMProvider.OPENROUTER:
        openrouter_key = (llm_config.api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if not openrouter_key:
            raise ValueError(
                "OPENROUTER provider selected but no API key found. "
                "Set OPENROUTER_API_KEY or pass llm.api_key."
            )
        if not openrouter_key.startswith("sk-or-"):
            raise ValueError(
                "OPENROUTER_API_KEY looks invalid (expected prefix `sk-or-`). "
                "You may be passing a non-OpenRouter key."
            )
        base_url = llm_config.base_url or "https://openrouter.ai/api/v1"
        # Force OpenAI-compatible client config to OpenRouter to avoid env collisions.
        kwargs["api_key"] = openrouter_key
        kwargs["openai_api_key"] = openrouter_key
        kwargs["base_url"] = base_url
        kwargs["openai_api_base"] = base_url
        os.environ["OPENAI_API_KEY"] = openrouter_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("Install `langchain-openai` to use OPENROUTER provider.") from exc
        return ChatOpenAI(model=llm_config.model, **kwargs)

    if llm_config.provider == LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError("Install `langchain-anthropic` to use ANTHROPIC provider.") from exc
        return ChatAnthropic(model=llm_config.model, **kwargs)

    if llm_config.provider == LLMProvider.GOOGLE:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError("Install `langchain-google-genai` to use GOOGLE provider.") from exc
        return ChatGoogleGenerativeAI(model=llm_config.model, **kwargs)

    if llm_config.provider == LLMProvider.OLLAMA:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError("Install `langchain-ollama` to use OLLAMA provider.") from exc
        return ChatOllama(model=llm_config.model, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")


async def analyze_query_node(
    state: AgentState,
    llm: BaseChatModel,
    allow_decomposition: bool,
    verbose: bool = False,
) -> dict[str, Any]:
    if verbose:
        print(f"\n[ANALYZE] {state['current_query']}")

    prompt = QUERY_ANALYSIS_PROMPT.format(query=state["current_query"])
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:
        _raise_llm_error_context(llm, exc)
    analysis = parse_json_response(_extract_text(response.content))

    analysis["original_query"] = state["current_query"]
    queries = _normalize_queries(
        candidate_queries=analysis.get("search_queries"),
        fallback_query=state["current_query"],
        allow_decomposition=allow_decomposition,
        max_queries=4,
    )
    analysis["search_queries"] = queries

    if verbose:
        print(f"  Planned queries: {queries}")

    return {
        "query_analysis": analysis,
        "search_queries": queries,
        "search_index": 0,
        "phase": AgentPhase.SEARCHING.value,
    }


async def search_node(
    state: AgentState,
    tool_factory: ToolFactory,
    max_total_results: int,
    deduplicate: bool,
    verbose: bool = False,
) -> dict[str, Any]:
    if verbose:
        print(f"\n[SEARCH] Executing {len(state['search_queries'])} query(ies)")

    all_results = list(state["search_results"])
    search_history = list(state["search_history"])
    seen_ids = {r.get("id") for r in all_results if r.get("id")}

    for query in state["search_queries"]:
        search_output = await tool_factory.search(query)
        added = 0

        for result in search_output["results"]:
            result_id = result.get("id")
            if deduplicate and result_id and result_id in seen_ids:
                continue
            if result_id:
                seen_ids.add(result_id)
            all_results.append(result)
            added += 1
            if len(all_results) >= max_total_results:
                break

        search_history.append(
            {
                "query": query,
                "results": search_output["results"],
                "new_results": added,
                "reasoning": f"Search iteration {state['iteration_count'] + 1}",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        if verbose:
            print(f"  Query: {query!r} -> {len(search_output['results'])} results ({added} added)")

        if len(all_results) >= max_total_results:
            if verbose:
                print(f"  Reached max_total_results={max_total_results}")
            break

    return {
        "search_results": all_results,
        "search_history": search_history,
        "search_queries": [],
        "phase": AgentPhase.EVALUATING.value,
    }


async def evaluate_context_node(
    state: AgentState,
    llm: BaseChatModel,
    min_confidence: float = 0.7,
    verbose: bool = False,
) -> dict[str, Any]:
    iteration = state["iteration_count"] + 1
    has_context = bool(state["search_results"])

    if not has_context:
        next_phase = (
            AgentPhase.SYNTHESIZING.value
            if iteration >= state["max_iterations"]
            else AgentPhase.SEARCHING.value
        )
        next_query = [] if next_phase == AgentPhase.SYNTHESIZING.value else [state["current_query"]]
        return {
            "context_evaluation": {
                "is_sufficient": False,
                "confidence": 0.0,
                "missing_information": "No relevant documents were retrieved.",
                "suggested_query": state["current_query"],
                "reasoning": "No context available after retrieval.",
            },
            "iteration_count": iteration,
            "phase": next_phase,
            "search_queries": next_query,
        }

    context_text = format_context_for_prompt(state["search_results"])
    history_text = format_search_history(state["search_history"])
    prompt = CONTEXT_EVALUATION_PROMPT.format(
        query=state["current_query"],
        context=context_text,
        search_history=history_text,
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:
        _raise_llm_error_context(llm, exc)
    evaluation = parse_json_response(_extract_text(response.content))

    confidence = _to_float(evaluation.get("confidence"), 0.0)
    is_sufficient = _to_bool(evaluation.get("is_sufficient"), False)
    suggested_query = evaluation.get("suggested_query")
    if not isinstance(suggested_query, str) or not suggested_query.strip():
        suggested_query = state["current_query"]
    suggested_query = suggested_query.strip()

    normalized = {
        "is_sufficient": is_sufficient,
        "confidence": confidence,
        "missing_information": evaluation.get("missing_information"),
        "suggested_query": suggested_query,
        "reasoning": evaluation.get("reasoning", "No evaluation provided"),
    }

    if is_sufficient and confidence >= min_confidence:
        next_phase = AgentPhase.SYNTHESIZING.value
        next_queries: list[str] = []
    elif iteration >= state["max_iterations"]:
        next_phase = AgentPhase.SYNTHESIZING.value
        next_queries = []
    else:
        next_phase = AgentPhase.SEARCHING.value
        next_queries = [suggested_query]

    if verbose:
        print(
            "[EVALUATE] "
            f"sufficient={is_sufficient} confidence={confidence:.2f} "
            f"next_phase={next_phase}"
        )

    return {
        "context_evaluation": normalized,
        "iteration_count": iteration,
        "phase": next_phase,
        "search_queries": next_queries,
    }


async def synthesize_answer_node(
    state: AgentState,
    llm: BaseChatModel,
    verbose: bool = False,
) -> dict[str, Any]:
    if not state["search_results"]:
        answer = (
            "I could not find relevant context in the collection to answer that question. "
            "Try rephrasing your query or ingesting more source documents."
        )
        return {
            "final_answer": answer,
            "citations": [],
            "phase": AgentPhase.COMPLETE.value,
            "messages": [HumanMessage(content=state["current_query"]), AIMessage(content=answer)],
        }

    context_text = format_context_for_prompt(state["search_results"])
    prompt = ANSWER_SYNTHESIS_PROMPT.format(
        query=state["current_query"],
        context=context_text,
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:
        _raise_llm_error_context(llm, exc)
    answer = _extract_text(response.content)

    citations: list[dict[str, Any]] = []
    for i, result in enumerate(state["search_results"], 1):
        content = result.get("content", "")
        citations.append(
            {
                "index": i,
                "content": (content[:200] + "...") if len(content) > 200 else content,
                "score": _to_float(result.get("score"), 0.0),
                "document_id": result.get("document_id"),
            }
        )

    if verbose:
        print(f"[SYNTHESIZE] generated answer with {len(citations)} citation(s)")

    return {
        "final_answer": answer,
        "citations": citations,
        "phase": AgentPhase.COMPLETE.value,
        "messages": [HumanMessage(content=state["current_query"]), AIMessage(content=answer)],
    }


def route_after_evaluation(state: AgentState) -> Literal["search", "synthesize"]:
    if state["phase"] == AgentPhase.SYNTHESIZING.value:
        return "synthesize"
    return "search"


class AgenticRAGAgent:
    """
    LangGraph-based agentic RAG agent.

    Example:
        config = AgentConfig(collection_id="my-collection")
        agent = await AgenticRAGAgent.create(config)
        result = await agent.run("What are the key design choices in this project?")
        print(result["answer"])
    """

    def __init__(
        self,
        config: AgentConfig,
        ragora_client: RagoraClient,
        llm: BaseChatModel,
        graph: StateGraph,
        checkpointer: Any | None = None,
        sqlite_conn: Any | None = None,
    ):
        self.config = config
        self.ragora_client = ragora_client
        self.llm = llm
        self.graph = graph
        self.checkpointer = checkpointer
        self._sqlite_conn = sqlite_conn
        self._compiled_graph = self.graph.compile(checkpointer=self.checkpointer)

    @classmethod
    async def create(
        cls,
        config: AgentConfig,
        ragora_client: RagoraClient | None = None,
    ) -> AgenticRAGAgent:
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")

        if ragora_client is None:
            api_key = config.ragora_api_key or os.environ.get("RAGORA_API_KEY")
            if not api_key:
                raise ValueError(
                    "RAGORA_API_KEY environment variable or config.ragora_api_key is required."
                )
            ragora_client = RagoraClient(api_key=api_key, base_url=config.ragora_base_url)

        llm = create_llm(config)

        checkpointer: Any = MemorySaver()
        sqlite_conn: Any | None = None
        if config.persistence_enabled and AsyncSqliteSaver:
            try:
                import aiosqlite

                sqlite_conn = await aiosqlite.connect(config.persistence_path)
                checkpointer = AsyncSqliteSaver(sqlite_conn)
                await checkpointer.setup()
            except Exception as exc:
                checkpointer = MemorySaver()
                if config.verbose:
                    print(f"[WARN] SQLite checkpoint unavailable; using memory checkpoint: {exc}")

        graph = cls._build_graph(config, ragora_client, llm)

        return cls(
            config=config,
            ragora_client=ragora_client,
            llm=llm,
            graph=graph,
            checkpointer=checkpointer,
            sqlite_conn=sqlite_conn,
        )

    @classmethod
    def _build_graph(
        cls,
        config: AgentConfig,
        ragora_client: RagoraClient,
        llm: BaseChatModel,
    ) -> StateGraph:
        tool_factory = ToolFactory(
            ragora_client=ragora_client,
            collection_id=config.collection_id,
            top_k=config.search.top_k,
            threshold=config.search.threshold,
        )

        async def analyze(state: AgentState) -> dict[str, Any]:
            return await analyze_query_node(
                state=state,
                llm=llm,
                allow_decomposition=config.enable_query_decomposition,
                verbose=config.verbose,
            )

        async def search(state: AgentState) -> dict[str, Any]:
            return await search_node(
                state=state,
                tool_factory=tool_factory,
                max_total_results=config.search.max_total_results,
                deduplicate=config.search.deduplicate,
                verbose=config.verbose,
            )

        async def evaluate(state: AgentState) -> dict[str, Any]:
            return await evaluate_context_node(
                state=state,
                llm=llm,
                min_confidence=config.min_confidence,
                verbose=config.verbose,
            )

        async def synthesize(state: AgentState) -> dict[str, Any]:
            return await synthesize_answer_node(state=state, llm=llm, verbose=config.verbose)

        builder = StateGraph(AgentState)
        builder.add_node("analyze", analyze)
        builder.add_node("search", search)
        builder.add_node("evaluate", evaluate)
        builder.add_node("synthesize", synthesize)

        builder.set_entry_point("analyze")
        builder.add_edge("analyze", "search")
        builder.add_edge("search", "evaluate")
        builder.add_conditional_edges(
            "evaluate",
            route_after_evaluation,
            {
                "search": "search",
                "synthesize": "synthesize",
            },
        )
        builder.add_edge("synthesize", END)
        return builder

    async def run(self, query: str, session_id: str | None = None) -> dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        initial_state = create_initial_state(
            query=query,
            session_id=session_id,
            collection_id=self.config.collection_id,
            model_name=self.config.llm.model,
            max_iterations=self.config.max_iterations,
        )

        final_state = await self._compiled_graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
        )

        return {
            "answer": final_state.get("final_answer", ""),
            "citations": final_state.get("citations", []),
            "search_history": final_state.get("search_history", []),
            "iteration_count": final_state.get("iteration_count", 0),
            "session_id": session_id,
        }

    async def stream(self, query: str, session_id: str | None = None) -> AsyncIterator[str]:
        result = await self.run(query, session_id)
        answer = result.get("answer", "")
        chunk_size = 80
        for i in range(0, len(answer), chunk_size):
            yield answer[i : i + chunk_size]
            await asyncio.sleep(0.02)

    async def close(self) -> None:
        if self.ragora_client:
            await self.ragora_client.close()
        if self._sqlite_conn is not None:
            await self._sqlite_conn.close()
            self._sqlite_conn = None


async def create_agent(
    collection_id: str,
    provider: str = "openrouter",
    model: str | None = None,
    **kwargs: Any,
) -> AgenticRAGAgent:
    """
    Convenience factory.

    Accepts AgentConfig kwargs plus LLMConfig kwargs:
    - AgentConfig: ragora_api_key, ragora_base_url, max_iterations, min_confidence,
      enable_query_decomposition, persistence_enabled, persistence_path, streaming_enabled, verbose
    - LLMConfig: temperature, max_tokens, api_key, base_url
    """
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "anthropic": LLMProvider.ANTHROPIC,
        "google": LLMProvider.GOOGLE,
        "ollama": LLMProvider.OLLAMA,
        "openrouter": LLMProvider.OPENROUTER,
    }
    llm_provider = provider_map.get(provider.lower())
    if llm_provider is None:
        supported = ", ".join(sorted(provider_map.keys()))
        raise ValueError(f"Unsupported provider {provider!r}. Supported values: {supported}")

    model_name = model or DEFAULT_MODELS[llm_provider]

    agent_field_names = {f.name for f in fields(AgentConfig)} - {"llm", "collection_id"}
    llm_field_names = {f.name for f in fields(LLMConfig)} - {"provider", "model"}

    agent_kwargs = {k: v for k, v in kwargs.items() if k in agent_field_names}
    llm_kwargs = {k: v for k, v in kwargs.items() if k in llm_field_names}
    unknown = sorted(set(kwargs) - set(agent_kwargs) - set(llm_kwargs))
    if unknown:
        raise ValueError(f"Unknown create_agent kwargs: {', '.join(unknown)}")

    config = AgentConfig(
        collection_id=collection_id,
        llm=LLMConfig(provider=llm_provider, model=model_name, **llm_kwargs),
        **agent_kwargs,
    )
    return await AgenticRAGAgent.create(config)
