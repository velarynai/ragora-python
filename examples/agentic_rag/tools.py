"""
Agent Tools

Tools that the agent can use to interact with the knowledge base.
These are LangGraph-compatible tools that wrap the Ragora SDK.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ragora import RagoraClient


# ============================================================================
# Tool Input/Output Schemas
# ============================================================================


class SearchInput(BaseModel):
    """Input for the search tool."""
    query: str = Field(..., description="The search query to find relevant documents")
    top_k: int = Field(5, description="Number of results to return")


class SearchOutput(BaseModel):
    """Output from the search tool."""
    query: str
    results: list[dict[str, Any]]
    total: int


class QueryAnalysisInput(BaseModel):
    """Input for query analysis."""
    query: str = Field(..., description="The user's question to analyze")


class QueryAnalysisOutput(BaseModel):
    """Output from query analysis."""
    original_query: str
    intent: str
    search_queries: list[str]
    complexity: str
    requires_multiple_searches: bool
    reasoning: str


# ============================================================================
# Tool Factory
# ============================================================================


class ToolFactory:
    """
    Factory for creating LangGraph-compatible tools.
    
    Tools are created with a reference to the Ragora client and configuration.
    """
    
    def __init__(
        self,
        ragora_client: RagoraClient,
        collection_id: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ):
        self.client = ragora_client
        self.collection_id = collection_id
        self.top_k = top_k
        self.threshold = threshold
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Search the knowledge base for relevant documents.
        
        Args:
            query: The search query
            top_k: Number of results (defaults to config)
            
        Returns:
            Search results with content and relevance scores
        """
        k = top_k or self.top_k
        
        response = await self.client.search(
            collection_id=self.collection_id,
            query=query,
            top_k=k,
            threshold=self.threshold,
        )
        
        results = [
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
                "document_id": r.document_id,
            }
            for r in response.results
        ]
        
        return {
            "query": query,
            "results": results,
            "total": len(results),
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
    ):
        """
        Generate a response using the Ragora chat API.
        
        This uses the collection's context for RAG-enhanced generation.
        """
        if stream:
            return self.client.chat_stream(
                collection_id=self.collection_id,
                messages=messages,
                temperature=temperature,
            )
        else:
            return await self.client.chat(
                collection_id=self.collection_id,
                messages=messages,
                temperature=temperature,
            )


# ============================================================================
# Prompt Templates for Tool Operations
# ============================================================================


QUERY_ANALYSIS_PROMPT = """You are an expert at analyzing user questions to determine the best search strategy.

Analyze this question and determine:
1. What is the user's main intent?
2. What search queries would find the most relevant information?
3. Is this a simple or complex question?
4. Does it require multiple searches to answer fully?

User Question: {query}

Respond with a JSON object:
{{
    "intent": "concise description of what the user wants to know",
    "search_queries": ["query1", "query2", "query3"],
    "complexity": "simple" | "moderate" | "complex",
    "requires_multiple_searches": true | false,
    "reasoning": "brief explanation of your analysis"
}}

Guidelines for search queries:
- Keep queries concise (2-8 words)
- Focus on key concepts and entities
- For complex questions, break into sub-questions
- Maximum 4 queries

Respond ONLY with the JSON object, no other text."""


CONTEXT_EVALUATION_PROMPT = """You are evaluating whether the gathered context is sufficient to answer a question.

User Question: {query}

Gathered Context:
{context}

Search History:
{search_history}

Evaluate:
1. Does the context contain enough information to fully answer the question?
2. What specific information is missing, if any?
3. What additional search query might fill the gap?

Respond with a JSON object:
{{
    "is_sufficient": true | false,
    "confidence": 0.0 to 1.0,
    "missing_information": "description of what's missing" | null,
    "suggested_query": "query to find missing info" | null,
    "reasoning": "brief explanation"
}}

Be conservative - only mark as sufficient if you're confident the context
can provide a complete, accurate answer.

Respond ONLY with the JSON object, no other text."""


ANSWER_SYNTHESIS_PROMPT = """You are a helpful assistant answering questions based on provided context.

User Question: {query}

Context (cite using [1], [2], etc.):
{context}

Instructions:
1. Answer the question using ONLY information from the provided context
2. Use citations [1], [2], etc. to reference specific sources
3. If the context doesn't fully answer the question, acknowledge the gaps
4. Be comprehensive but concise
5. Structure your answer clearly with paragraphs or bullet points if helpful

If you cannot answer based on the context, say so clearly and explain what information would be needed.

Your Answer:"""


# ============================================================================
# Utility Functions
# ============================================================================


def format_context_for_prompt(results: list[dict[str, Any]]) -> str:
    """Format search results as numbered context for the LLM."""
    if not results:
        return "No relevant context found."
    
    parts = []
    for i, result in enumerate(results, 1):
        content = result.get("content", "")
        score = result.get("score", 0)
        parts.append(f"[{i}] (relevance: {score:.2f})\n{content}")
    
    return "\n\n".join(parts)


def format_search_history(history: list[dict[str, Any]]) -> str:
    """Format search history for the evaluation prompt."""
    if not history:
        return "No previous searches."
    
    parts = []
    for i, search in enumerate(history, 1):
        query = search.get("query", "")
        num_results = len(search.get("results", []))
        parts.append(f"{i}. Query: \"{query}\" → {num_results} results")
    
    return "\n".join(parts)


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling common issues."""
    # Remove markdown code blocks if present
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        
        # Return a default structure
        return {
            "error": "Failed to parse JSON response",
            "raw_text": text[:500],
        }
