# langchain-calculator-agent

A LangChain calculator agent that handles basic arithmetic in natural language using OpenAI tool-calling.

## Project Overview

This project demonstrates the standard agent pattern:

- LLM (reasoning)
- Tools (add/subtract/multiply/divide)
- Decision loop (agent executor)

### In Scope

- Digits (integers/decimals), negative numbers
- `plus`, `minus`, `times`, `multiplied by`, `divided by`
- Parentheses for explicit order
- Minor typos and spelled numbers when intent is clear (LLM interpretation)

### Out of Scope

- General knowledge questions
- Advanced math (e.g., trig, logarithms)

## Architecture Flow

1. User submits a query.
2. LangChain agent is always attempted first and selects arithmetic tools.
3. If LLM execution fails and fallback is enabled, strict pure expressions are evaluated safely.
4. Final output returns only the numeric result or a guardrail message.

## Repository Structure

```text
langchain-calculator-agent/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── agent.py
│   └── tools/
│       ├── __init__.py
│       └── arithmetic.py
├── tests/
│   ├── __init__.py
│   └── test_agent.py
├── docs/
│   └── notes.md
├── .env.example
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your key in `.env`:

```bash
OPENAI_API_KEY="your_api_key_here"
```

## Usage

One-shot mode:

```bash
python -m src.main --query "What is (12 plus 3) times 4 minus 10?"
```

Interactive mode:

```bash
python -m src.main
```

Enable deterministic fallback safety net:

```bash
python -m src.main --fallback --query "2 + 2 * (10 - 3)"
```

Or via environment variable:

```bash
DETERMINISTIC_FALLBACK=1 python -m src.main --query "2 + 2 * (10 - 3)"
```

Fallback policy:

- LLM execution is always attempted first.
- Fallback is disabled by default.
- Fallback runs only when LLM execution fails and input is a strict pure expression (`0-9`, spaces, `+ - * / ( ) .`).

## Tool Definitions

- `add(a, b) -> float`
- `subtract(a, b) -> float`
- `multiply(a, b) -> float`
- `divide(a, b) -> float` (raises error for division by zero)

## Required Examples

- `What is (12 plus 3) times 4 minus 10?` -> `50.0`
- `Compute 100 divided by (5 times 4) plus 7.` -> `12.0`
- `What is -8 times 3 plus 2 divided by 4?` -> `-23.5`
- `What is 10 divided by 0?` -> tool failure message
- `What is the capital of France?` -> out-of-scope message

## Testing

```bash
python -m pytest -q tests/test_agent.py
```

## Known Limitations

- LLM output quality depends on model behavior.
- Fallback does not parse natural-language math; it only supports strict expressions.

## Future Enhancements

- Better ambiguity detection for natural language phrasing
- Rich tracing/logging (for example with LangSmith)
- Stronger parser for broader natural language math support
