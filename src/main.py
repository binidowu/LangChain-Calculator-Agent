"""CLI entrypoint for the calculator agent."""

from __future__ import annotations

import argparse

from .agent import CalculatorAgent, LLM_SETUP_MESSAGE


def build_parser() -> argparse.ArgumentParser:
    """Build parser for one-shot and interactive modes."""
    parser = argparse.ArgumentParser(description="Run the LangChain calculator agent.")
    parser.add_argument(
        "--query",
        type=str,
        help="Optional one-shot query. If omitted, interactive mode is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name used for tool-calling execution.",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Enable deterministic fallback only when LLM execution fails.",
    )
    return parser


def main() -> None:
    """Run one-shot evaluation or an interactive loop."""
    args = build_parser().parse_args()

    try:
        agent = CalculatorAgent(model=args.model, deterministic_fallback=args.fallback)
    except RuntimeError as exc:
        print(str(exc) or LLM_SETUP_MESSAGE)
        raise SystemExit(1)

    if args.query:
        print(agent.ask(args.query))
        return

    print("Hello! I am the Calculator Agent. Ask me arithmetic questions.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        print(agent.ask(user_input))


if __name__ == "__main__":
    main()
