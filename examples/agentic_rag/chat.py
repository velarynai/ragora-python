#!/usr/bin/env python3
"""
Agentic RAG Chat CLI

Interactive chat powered by Ragora's agent system.
All RAG logic (search, memory, compaction, tool calls) is handled server-side.

Usage:
    # Create a new agent for a collection
    python -m examples.agentic_rag.chat --collection-id your-collection

    # Connect to an existing agent
    python -m examples.agentic_rag.chat --agent-id your-agent-id

    # Stream responses
    python -m examples.agentic_rag.chat --collection-id your-collection --stream
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .agent import AgenticRAGAgent


# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


def colored(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{''.join(codes)}{text}{Colors.RESET}"


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

class ChatInterface:
    COMMANDS = {
        "/help": "Show available commands",
        "/quit": "Exit the chat",
        "/new": "Start a new conversation",
        "/sessions": "List conversation sessions",
        "/history": "Show messages in current session",
        "/stream": "Toggle streaming mode",
    }

    def __init__(self, agent: AgenticRAGAgent, stream: bool = False):
        self.agent = agent
        self.stream = stream

    def print_banner(self) -> None:
        print()
        print(colored("=== Ragora Agentic RAG Chat ===", Colors.CYAN, Colors.BOLD))
        print(colored(f"  Agent:   {self.agent.agent_id}", Colors.DIM))
        if self.agent.session_id:
            print(colored(f"  Session: {self.agent.session_id}", Colors.DIM))
        print(colored("  Type /help for commands, /quit to exit", Colors.DIM))
        print()

    def print_help(self) -> None:
        print()
        print(colored("Commands:", Colors.BOLD))
        for cmd, desc in self.COMMANDS.items():
            print(f"  {colored(cmd, Colors.YELLOW):<20} {desc}")
        print()

    async def handle_command(self, command: str) -> bool:
        """Handle a slash command. Returns False to quit."""
        cmd = command.lower().strip()

        if cmd in ("/quit", "/exit"):
            return False

        if cmd == "/help":
            self.print_help()

        elif cmd == "/new":
            self.agent.new_session()
            print(colored("\nStarted new conversation.\n", Colors.GREEN))

        elif cmd == "/sessions":
            await self._show_sessions()

        elif cmd == "/history":
            await self._show_history()

        elif cmd == "/stream":
            self.stream = not self.stream
            mode = "ON" if self.stream else "OFF"
            print(colored(f"\nStreaming: {mode}\n", Colors.YELLOW))

        else:
            print(colored(
                f"Unknown command: {command}. Type /help for help.", Colors.RED
            ))

        return True

    async def _show_sessions(self) -> None:
        try:
            result = await self.agent.list_sessions()
            if not result.sessions:
                print(colored("\nNo sessions found.\n", Colors.DIM))
                return
            print(colored(f"\n{len(result.sessions)} session(s):", Colors.BOLD))
            for s in result.sessions:
                status_color = Colors.GREEN if s.status == "open" else Colors.DIM
                print(
                    f"  {s.id[:12]}... "
                    f"[{colored(s.status, status_color)}] "
                    f"{s.message_count} messages"
                )
            print()
        except Exception as e:
            print(colored(f"\nError listing sessions: {e}\n", Colors.RED))

    async def _show_history(self) -> None:
        if not self.agent.session_id:
            print(colored("\nNo active session. Send a message first.\n", Colors.DIM))
            return
        try:
            detail = await self.agent.get_session(self.agent.session_id)
            if not detail.messages:
                print(colored("\nNo messages in session.\n", Colors.DIM))
                return
            print()
            for msg in detail.messages:
                role_color = Colors.GREEN if msg.role == "user" else Colors.CYAN
                print(f"  {colored(msg.role, role_color, Colors.BOLD)}: {msg.content[:200]}")
            print()
        except Exception as e:
            print(colored(f"\nError: {e}\n", Colors.RED))

    async def process_query(self, query: str) -> None:
        print()
        try:
            if self.stream:
                print(colored("Assistant:", Colors.BOLD))
                async for chunk in self.agent.chat_stream(query):
                    if chunk["content"]:
                        print(chunk["content"], end="", flush=True)
                print("\n")
            else:
                print(colored("Thinking...", Colors.DIM), end="\r")
                result = await self.agent.chat(query)
                print("\033[2K", end="\r")  # clear line
                print(colored("Assistant:", Colors.BOLD))
                print(result["message"])

                citations = result.get("citations") or []
                if citations:
                    print(colored(f"\n  [{len(citations)} source(s)]", Colors.DIM))
                print()
        except Exception as e:
            print(colored(f"\nError: {e}\n", Colors.RED))

    async def run(self) -> None:
        self.print_banner()
        while True:
            try:
                user_input = input(colored("You: ", Colors.GREEN)).strip()
                if not user_input:
                    continue
                if user_input.startswith("/"):
                    if not await self.handle_command(user_input):
                        break
                    continue
                await self.process_query(user_input)
            except KeyboardInterrupt:
                print(colored("\n\nType /quit to exit.\n", Colors.YELLOW))
            except EOFError:
                break

        print(colored("\nGoodbye!\n", Colors.CYAN))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ragora Agentic RAG Chat",
        epilog="""
Examples:
  python -m examples.agentic_rag.chat --collection-id my-collection
  python -m examples.agentic_rag.chat --agent-id existing-agent-id --stream

Environment Variables:
  RAGORA_API_KEY         Your Ragora API key
  RAGORA_BASE_URL        API base URL (default: https://api.ragora.app)
  RAGORA_COLLECTION_ID   Default collection ID
  RAGORA_AGENT_ID        Default agent ID
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--collection-id", "-c",
        default=os.environ.get("RAGORA_COLLECTION_ID"),
        help="Collection ID to create an agent for",
    )
    parser.add_argument(
        "--agent-id", "-a",
        default=os.environ.get("RAGORA_AGENT_ID"),
        help="Existing agent ID to connect to",
    )
    parser.add_argument(
        "--name", "-n",
        default="RAG Agent",
        help="Agent name when creating a new agent (default: 'RAG Agent')",
    )
    parser.add_argument(
        "--system-prompt",
        help="Custom system prompt for the agent",
    )
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Enable streaming responses",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if not args.agent_id and not args.collection_id:
        print(colored(
            "Error: --collection-id or --agent-id is required "
            "(or set RAGORA_COLLECTION_ID / RAGORA_AGENT_ID)",
            Colors.RED,
        ))
        sys.exit(1)

    try:
        agent = await AgenticRAGAgent.create(
            collection_id=args.collection_id,
            name=args.name,
            system_prompt=args.system_prompt,
            agent_id=args.agent_id,
        )
    except Exception as e:
        print(colored(f"Error creating agent: {e}", Colors.RED))
        sys.exit(1)

    try:
        chat = ChatInterface(agent, stream=args.stream)
        await chat.run()
    finally:
        await agent.close()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
