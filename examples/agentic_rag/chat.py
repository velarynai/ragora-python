#!/usr/bin/env python3
"""
Agentic RAG - Interactive Chat CLI

A full-featured chat interface for the agentic RAG system.
Supports:
- Multiple LLM providers (OpenAI, Anthropic, Ollama, etc.)
- Streaming responses
- Conversation persistence
- Session management
- Verbose mode for debugging

Usage:
    python -m examples.agentic_rag.chat --collection-id your-collection

    # With specific provider
    python -m examples.agentic_rag.chat --collection-id your-collection --provider anthropic --model claude-3-5-sonnet

    # With Ollama (local)
    python -m examples.agentic_rag.chat --collection-id your-collection --provider ollama --model llama3.2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from typing import Optional

from .agent import AgenticRAGAgent
from .config import AgentConfig, LLMConfig, LLMProvider


# ============================================================================
# ANSI Colors
# ============================================================================


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"


def colored(text: str, *codes: str) -> str:
    """Apply color codes to text."""
    if not sys.stdout.isatty():
        return text
    return f"{''.join(codes)}{text}{Colors.RESET}"


# ============================================================================
# Chat Interface
# ============================================================================


class ChatInterface:
    """Interactive chat interface for the agentic RAG system."""
    
    COMMANDS = {
        "/help": "Show this help message",
        "/quit": "Exit the chat",
        "/new": "Start a new conversation",
        "/verbose": "Toggle verbose mode",
        "/history": "Show search history from last query",
        "/citations": "Show citations from last answer",
        "/config": "Show current configuration",
    }
    
    def __init__(self, agent: AgenticRAGAgent, session_id: Optional[str] = None):
        self.agent = agent
        self.session_id = session_id or str(uuid.uuid4())
        self.last_result: Optional[dict] = None
        self.verbose = agent.config.verbose
    
    def print_banner(self):
        """Print welcome banner."""
        print()
        print(colored("╔══════════════════════════════════════════════════════════════╗", Colors.CYAN))
        print(colored("║", Colors.CYAN) + colored("             🤖 AGENTIC RAG CHAT                              ", Colors.BOLD) + colored("║", Colors.CYAN))
        print(colored("║", Colors.CYAN) + "                                                              " + colored("║", Colors.CYAN))
        print(colored("║", Colors.CYAN) + f"  Provider: {self.agent.config.llm.provider.value:<12}  Model: {self.agent.config.llm.model:<20}" + colored("║", Colors.CYAN))
        print(colored("║", Colors.CYAN) + f"  Collection: {self.agent.config.collection_id:<48}" + colored("║", Colors.CYAN))
        print(colored("║", Colors.CYAN) + "                                                              " + colored("║", Colors.CYAN))
        print(colored("║", Colors.CYAN) + "  Type " + colored("/help", Colors.YELLOW) + " for commands, " + colored("/quit", Colors.YELLOW) + " to exit                     " + colored("║", Colors.CYAN))
        print(colored("╚══════════════════════════════════════════════════════════════╝", Colors.CYAN))
        print()
    
    def print_help(self):
        """Print help message."""
        print()
        print(colored("Available Commands:", Colors.BOLD))
        for cmd, desc in self.COMMANDS.items():
            print(f"  {colored(cmd, Colors.YELLOW):<15} {desc}")
        print()
    
    def print_history(self):
        """Print search history from last query."""
        if not self.last_result:
            print(colored("No search history yet. Ask a question first!", Colors.DIM))
            return
        
        history = self.last_result.get("search_history", [])
        if not history:
            print(colored("No searches were performed.", Colors.DIM))
            return
        
        print()
        print(colored("📚 Search History:", Colors.BOLD))
        for i, search in enumerate(history, 1):
            print(f"\n  {colored(str(i) + '.', Colors.CYAN)} Query: {colored(search['query'], Colors.GREEN)}")
            print(f"     Results: {len(search['results'])} documents")
            if search.get("new_results") is not None:
                print(f"     New: {search['new_results']} unique documents")
        print()
    
    def print_citations(self):
        """Print citations from last answer."""
        if not self.last_result:
            print(colored("No citations yet. Ask a question first!", Colors.DIM))
            return
        
        citations = self.last_result.get("citations", [])
        if not citations:
            print(colored("No citations available.", Colors.DIM))
            return
        
        print()
        print(colored("📄 Citations:", Colors.BOLD))
        for cite in citations[:10]:  # Limit to 10
            print(f"\n  {colored('[' + str(cite['index']) + ']', Colors.CYAN)} (relevance: {cite['score']:.2f})")
            content = cite['content'][:150] + "..." if len(cite['content']) > 150 else cite['content']
            print(f"     {content}")
        print()
    
    def print_config(self):
        """Print current configuration."""
        config = self.agent.config
        print()
        print(colored("⚙️  Configuration:", Colors.BOLD))
        print(f"  Collection ID: {colored(config.collection_id, Colors.GREEN)}")
        print(f"  LLM Provider:  {colored(config.llm.provider.value, Colors.CYAN)}")
        print(f"  Model:         {colored(config.llm.model, Colors.CYAN)}")
        print(f"  Temperature:   {config.llm.temperature}")
        print(f"  Max Iterations:{config.max_iterations}")
        print(f"  Search Top-K:  {config.search.top_k}")
        print(f"  Verbose:       {colored('ON' if self.verbose else 'OFF', Colors.YELLOW)}")
        print(f"  Session ID:    {colored(self.session_id[:8] + '...', Colors.DIM)}")
        print()
    
    async def handle_command(self, command: str) -> bool:
        """
        Handle a command.
        
        Returns True if chat should continue, False to exit.
        """
        cmd = command.lower().strip()
        
        if cmd == "/quit" or cmd == "/exit":
            return False
        
        if cmd == "/help":
            self.print_help()
        
        elif cmd == "/new":
            self.session_id = str(uuid.uuid4())
            self.last_result = None
            print(colored(f"\n🔄 Started new conversation (session: {self.session_id[:8]}...)\n", Colors.GREEN))
        
        elif cmd == "/verbose":
            self.verbose = not self.verbose
            self.agent.config.verbose = self.verbose
            print(colored(f"\n{'🔊' if self.verbose else '🔇'} Verbose mode: {'ON' if self.verbose else 'OFF'}\n", Colors.YELLOW))
        
        elif cmd == "/history":
            self.print_history()
        
        elif cmd == "/citations":
            self.print_citations()
        
        elif cmd == "/config":
            self.print_config()
        
        else:
            print(colored(f"Unknown command: {command}. Type /help for available commands.", Colors.RED))
        
        return True
    
    async def process_query(self, query: str):
        """Process a user query and display the response."""
        print()
        print(colored("🔍 Processing...", Colors.DIM))
        
        try:
            # Run the agent
            result = await self.agent.run(query, self.session_id)
            self.last_result = result
            
            # Clear the "Processing..." line
            print("\033[1A\033[2K", end="")  # Move up and clear line
            
            # Print the answer
            print(colored("🤖 Assistant:", Colors.BOLD))
            print()
            print(result["answer"])
            print()
            
            # Print summary
            num_citations = len(result.get("citations", []))
            iterations = result.get("iteration_count", 1)
            
            summary_parts = [
                colored(f"📚 {num_citations} sources", Colors.DIM),
                colored(f"🔄 {iterations} iteration(s)", Colors.DIM),
            ]
            print(" • ".join(summary_parts))
            print()
            
        except Exception as e:
            print(colored(f"\n❌ Error: {e}\n", Colors.RED))
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    async def run(self):
        """Run the interactive chat loop."""
        self.print_banner()
        
        while True:
            try:
                # Get user input
                user_input = input(colored("📝 You: ", Colors.GREEN)).strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    should_continue = await self.handle_command(user_input)
                    if not should_continue:
                        break
                    continue
                
                # Process query
                await self.process_query(user_input)
                
            except KeyboardInterrupt:
                print(colored("\n\nInterrupted. Type /quit to exit.\n", Colors.YELLOW))
            
            except EOFError:
                break
        
        print(colored("\n👋 Goodbye!\n", Colors.CYAN))


# ============================================================================
# CLI Entry Point
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG Chat - Intelligent retrieval-augmented generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using OpenRouter (default)
  python -m examples.agentic_rag.chat --collection-id my-collection

  # Using Anthropic Claude
  python -m examples.agentic_rag.chat --collection-id my-collection \\
      --provider anthropic --model claude-opus-4.6

  # Using local Ollama
  python -m examples.agentic_rag.chat --collection-id my-collection \\
      --provider ollama --model llama3.2 --ollama-url http://localhost:11434

Environment Variables:
  RAGORA_API_KEY       Your Ragora API key
  RAGORA_COLLECTION_ID Default collection ID
  OPENAI_API_KEY       OpenAI API key (for OpenAI provider)
  ANTHROPIC_API_KEY    Anthropic API key (for Anthropic provider)
        """,
    )
    
    # Required
    parser.add_argument(
        "--collection-id", "-c",
        default=os.environ.get("RAGORA_COLLECTION_ID"),
        help="Ragora collection ID (or set RAGORA_COLLECTION_ID)",
    )
    
    # LLM settings
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "anthropic", "google", "ollama", "openrouter"],
        default="openrouter",
        help="LLM provider (default: openrouter)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name (defaults based on provider)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    
    # Ollama specific
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    
    # Agent behavior
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum search refinement iterations (default: 3)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results per query (default: 5)",
    )
    
    # UI
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--no-persistence",
        action="store_true",
        help="Disable conversation persistence",
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate collection ID
    if not args.collection_id:
        print(colored("Error: --collection-id is required (or set RAGORA_COLLECTION_ID)", Colors.RED))
        sys.exit(1)
    
    # Build configuration
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "anthropic": LLMProvider.ANTHROPIC,
        "google": LLMProvider.GOOGLE,
        "ollama": LLMProvider.OLLAMA,
        "openrouter": LLMProvider.OPENROUTER,
    }
    
    llm_provider = provider_map[args.provider]
    
    # Default models
    default_models = {
        LLMProvider.OPENAI: "gpt-5.3-codex",
        LLMProvider.ANTHROPIC: "claude-opus-4.6",
        LLMProvider.GOOGLE: "gemini-3-pro",
        LLMProvider.OLLAMA: "llama-4-70b",
        LLMProvider.OPENROUTER: "google/gemini-3-flash-preview",
    }
    
    model = args.model or default_models[llm_provider]
    
    llm_config = LLMConfig(
        provider=llm_provider,
        model=model,
        temperature=args.temperature,
    )
    
    if llm_provider == LLMProvider.OLLAMA:
        llm_config.base_url = args.ollama_url
    
    config = AgentConfig(
        collection_id=args.collection_id,
        llm=llm_config,
        max_iterations=args.max_iterations,
        persistence_enabled=not args.no_persistence,
        verbose=args.verbose,
    )
    config.search.top_k = args.top_k
    
    # Create agent
    try:
        agent = await AgenticRAGAgent.create(config)
    except Exception as e:
        print(colored(f"Error creating agent: {e}", Colors.RED))
        sys.exit(1)
    
    # Run chat
    try:
        chat = ChatInterface(agent)
        await chat.run()
    finally:
        await agent.close()


def run():
    """Synchronous entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
