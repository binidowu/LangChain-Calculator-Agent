"""Arithmetic tool definitions used by the calculator agent.

The functions in this module are intentionally small and strict because they are
invoked by both:
1. The LangChain tool-calling agent.
2. The local deterministic fallback evaluator used in tests.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

TOOL_FAILURE_MESSAGE = (
    "I couldn't compute that due to invalid input "
    "(e.g., division by zero or non-numeric values)."
)


def _to_float(value: Any) -> float:
    """Convert a value to float and raise a controlled error if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Non-numeric value received.") from exc


@tool(
    "add",
    return_direct=False, # This means the agent will use the tool's return value in its reasoning rather than returning it directly to the user
)
def add(a: float, b: float) -> float:
    """Add exactly two numbers. Use this for operations phrased as plus/add."""
    left = _to_float(a)
    right = _to_float(b)
    return left + right


@tool(
    "subtract",
    return_direct=False,
)
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first. Use this for minus/subtract."""
    left = _to_float(a)
    right = _to_float(b)
    return left - right


@tool(
    "multiply",
    return_direct=False,
)
def multiply(a: float, b: float) -> float:
    """Multiply exactly two numbers. Use this for times/multiplied by."""
    left = _to_float(a)
    right = _to_float(b)
    return left * right


@tool(
    "divide",
    return_direct=False,
)
def divide(a: float, b: float) -> float:
    """Divide the first number by the second. Never call with zero as divisor."""
    left = _to_float(a)
    right = _to_float(b)
    if right == 0:
        raise ValueError("Division by zero is not allowed.")
    return left / right


def get_arithmetic_tools() -> list:
    """Return the tool list in a stable order for predictable agent behavior."""
    return [add, subtract, multiply, divide]
