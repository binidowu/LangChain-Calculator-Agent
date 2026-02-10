"""Behavior tests for the calculator agent.

These tests mock the LangChain runtime so they can verify output behavior
without making external network calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agent import (
    OUT_OF_SCOPE_MESSAGE,
    TOOL_FAILURE_MESSAGE,
    CalculatorAgent,
    _LangChainRuntime,
)


class _FakeExecutor:
    """Minimal executor stub that mirrors AgentExecutor.invoke contract."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses

    def invoke(self, payload: dict[str, Any]) -> dict[str, str]:
        query = str(payload.get("input", ""))
        if query not in self._responses:
            raise AssertionError(f"Unexpected query in test stub: {query}")
        return {"output": self._responses[query]}


@pytest.fixture()
def agent(monkeypatch: pytest.MonkeyPatch) -> CalculatorAgent:
    """Build an agent with a mocked LangChain runtime and fake API key."""
    responses = {
        "What is (12 plus 3) times 4 minus 10?": "50.0",
        "Compute 100 divided by (5 times 4) plus 7.": "12.0",
        "What is -8 times 3 plus 2 divided by 4?": "-23.5",
        "What is 10 divided by 0?": TOOL_FAILURE_MESSAGE,
        "What is the capital of France?": OUT_OF_SCOPE_MESSAGE,
        "What is seventeen plus 3?": "20",
        "Compute 25 times 34 and then plus 17": "867",
    }

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        CalculatorAgent,
        "_build_langchain_runtime",
        staticmethod(lambda **_: _LangChainRuntime(executor=_FakeExecutor(responses))),
    )
    return CalculatorAgent()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is (12 plus 3) times 4 minus 10?", "50.0"),
        ("Compute 100 divided by (5 times 4) plus 7.", "12.0"),
        ("What is -8 times 3 plus 2 divided by 4?", "-23.5"),
    ],
)
def test_happy_path_queries(agent: CalculatorAgent, query: str, expected: str) -> None:
    """Validate required successful arithmetic flows."""
    assert agent.ask(query) == expected


def test_division_by_zero(agent: CalculatorAgent) -> None:
    """Division by zero should return the standardized tool-failure response."""
    assert agent.ask("What is 10 divided by 0?") == TOOL_FAILURE_MESSAGE


def test_out_of_scope_question(agent: CalculatorAgent) -> None:
    """General knowledge queries must trigger the scope refusal guardrail."""
    assert agent.ask("What is the capital of France?") == OUT_OF_SCOPE_MESSAGE


def test_spelled_numbers_can_be_interpreted(agent: CalculatorAgent) -> None:
    """LLM-first path can interpret spelled numbers when intent is clear."""
    assert agent.ask("What is seventeen plus 3?") == "20.0"


def test_sequencing_phrase_can_be_interpreted(agent: CalculatorAgent) -> None:
    """LLM-first path can interpret sequencing language when it is clear."""
    assert agent.ask("Compute 25 times 34 and then plus 17") == "867.0"


def test_fallback_uses_strict_expression_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback should compute only strict pure expressions after LLM failure."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        CalculatorAgent,
        "_build_langchain_runtime",
        staticmethod(lambda **_: _LangChainRuntime(executor=_FakeExecutor({}))),
    )

    agent = CalculatorAgent(deterministic_fallback=True)
    monkeypatch.setattr(agent, "_run_langchain", lambda _: (_ for _ in ()).throw(RuntimeError("boom")))

    assert agent.ask("2 + 2 * (10 - 3)") == "16.0"
    # Natural language is not a strict regex match, so fallback must not run.
    assert agent.ask("What is 2 plus 2?") == TOOL_FAILURE_MESSAGE


def test_fallback_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without fallback enabled, LLM failure should return tool failure message."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        CalculatorAgent,
        "_build_langchain_runtime",
        staticmethod(lambda **_: _LangChainRuntime(executor=_FakeExecutor({}))),
    )
    agent = CalculatorAgent()
    monkeypatch.setattr(agent, "_run_langchain", lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    assert agent.ask("2 + 2") == TOOL_FAILURE_MESSAGE
