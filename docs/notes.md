# Notes

## Key Learnings

- Agent architecture is easiest to reason about as `LLM + Tools + Decision Loop`.
- Strict tool descriptions reduce incorrect tool selection by the LLM.
- Fast local guardrails (scope checks and spelled-number checks) improve reliability.
- A deterministic fallback evaluator is useful for repeatable offline tests.

## Architecture Summary

1. User query enters `src/main.py`.
2. `src/agent.py` applies guardrails.
3. If enabled and available, LangChain executes tool calls via OpenAI.
4. Otherwise, the deterministic evaluator parses and computes arithmetic safely.
5. Final user-visible output is a numeric result or a required guardrail message.
