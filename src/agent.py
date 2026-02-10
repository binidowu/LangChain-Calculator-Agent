"""Calculator agent orchestration.

Primary policy:
- Always run the LLM tool-calling agent first.
- Deterministic fallback is optional and disabled by default.
- Fallback is only used when the LLM path fails and input is a strict
  arithmetic expression.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from .tools.arithmetic import (
    TOOL_FAILURE_MESSAGE,
    add,
    divide,
    get_arithmetic_tools,
    multiply,
    subtract,
)

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with basic arithmetic (add/subtract/multiply/divide). "
    "Please provide a math expression."
)
LLM_SETUP_MESSAGE = (
    "LLM runtime is unavailable. Set OPENAI_API_KEY and install required "
    "dependencies (`langchain`, `langchain-openai`, `python-dotenv`)."
)


@dataclass
class _LangChainRuntime:
    """Container for LangChain executor state."""

    executor: Any


class CalculatorAgent:
    """Arithmetic-focused agent with optional deterministic safety-net fallback."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        deterministic_fallback: bool | None = None,
    ) -> None:
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(LLM_SETUP_MESSAGE)

        runtime = self._build_langchain_runtime(model=model, temperature=temperature)
        if runtime is None:
            raise RuntimeError(LLM_SETUP_MESSAGE)

        self._runtime = runtime
        if deterministic_fallback is None:
            deterministic_fallback = _is_truthy_env(os.getenv("DETERMINISTIC_FALLBACK"))
        self._deterministic_fallback = deterministic_fallback

    def ask(self, query: str) -> str:
        """Process one query and return a single user-facing response."""
        clean_query = query.strip()
        if not clean_query:
            return OUT_OF_SCOPE_MESSAGE

        # LLM is always first; this ensures your agent integration is always exercised.
        try:
            llm_output = self._run_langchain(clean_query)
            parsed = _extract_numeric_answer(llm_output)
            if parsed is not None:
                return _format_numeric(parsed)
            if llm_output:
                return llm_output
            return TOOL_FAILURE_MESSAGE
        except Exception:
            # Fallback is opt-in and only for strict numeric/operator expressions.
            if self._deterministic_fallback and _is_strict_pure_expression(clean_query):
                try:
                    return _format_numeric(_evaluate_pure_expression(clean_query))
                except Exception:
                    return TOOL_FAILURE_MESSAGE
            return TOOL_FAILURE_MESSAGE

    def _run_langchain(self, query: str) -> str:
        """Invoke the LangChain executor and extract the output string."""
        result = self._runtime.executor.invoke({"input": query})
        return str(result.get("output", "")).strip()

    @staticmethod
    def _build_langchain_runtime(*, model: str, temperature: float) -> _LangChainRuntime | None:
        """Build the LangChain runtime for OpenAI tool-calling."""
        try:
            from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_openai import ChatOpenAI
        except ImportError:
            return None

        tools = get_arithmetic_tools()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a strict arithmetic calculator agent. "
                    "Use tools to solve arithmetic. "
                    "You may interpret minor typos and spelled numbers when intent is clear. "
                    f"If the request is out of arithmetic scope, respond exactly: {OUT_OF_SCOPE_MESSAGE} "
                    "For successful calculations, return only the final numeric result.",
                ),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        llm = ChatOpenAI(model=model, temperature=temperature)
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
        )
        return _LangChainRuntime(executor=executor)


def _extract_numeric_answer(text: str) -> float | None:
    """Extract a numeric value from model output when present."""
    candidate = text.strip()
    try:
        return float(candidate)
    except ValueError:
        pass

    numbers = re.findall(r"-?\d+(?:\.\d+)?", candidate)
    if numbers:
        return float(numbers[-1])
    return None


def _format_numeric(value: float) -> str:
    """Normalize numeric output to a float-like string format."""
    return str(float(value))


def _is_truthy_env(value: str | None) -> bool:
    """Parse common truthy environment variable values."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_strict_pure_expression(query: str) -> bool:
    """Allow fallback only for expressions containing numeric/operator symbols."""
    if not re.fullmatch(r"[0-9\s\+\-\*/\(\)\.]+", query):
        return False
    has_digit = bool(re.search(r"\d", query))
    has_operator = bool(re.search(r"[\+\-\*/]", query))
    return has_digit and has_operator


def _evaluate_pure_expression(expression: str) -> float:
    """Safely evaluate a strict arithmetic expression via shared tool logic."""
    tree = ast.parse(expression, mode="eval")
    return _eval_ast(tree)


def _eval_ast(node: ast.AST) -> float:
    """Evaluate a restricted AST to prevent arbitrary code execution."""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return float(add.func(left, right))
        if isinstance(node.op, ast.Sub):
            return float(subtract.func(left, right))
        if isinstance(node.op, ast.Mult):
            return float(multiply.func(left, right))
        if isinstance(node.op, ast.Div):
            return float(divide.func(left, right))

    raise ValueError("Unsupported expression.")
