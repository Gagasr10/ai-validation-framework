# AI Model Validation Framework

> A production-ready Python + pytest framework for validating AI model outputs —
> accuracy, prompt injection security, LLM-as-Judge scoring, performance, and flakiness handling.

Two real LLMs in the loop: **Claude Haiku** (system under test) evaluated by both
**Claude Haiku** and **GPT-4o-mini** acting as independent judges.  
Falls back to a **deterministic mock** automatically when no API key is set —
CI/CD runs with zero cost and zero flakiness.

[![AI Model Validation](https://github.com/Gagasr10/ai-validation-framework/actions/workflows/ai-validation.yml/badge.svg)](https://github.com/Gagasr10/ai-validation-framework/actions)

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Test Suite (pytest)            │
│                                             │
│  recommend_recipe()   ←  Claude Haiku       │
│         │                (System Under Test)│
│         ▼                                   │
│  judge_response()     ←  Claude Haiku       │
│         │                (Judge 1)          │
│         ▼                                   │
│  judge_response()     ←  GPT-4o-mini        │
│         │                (Judge 2)          │
│         ▼                                   │
│  Dual-judge consensus check                 │
└─────────────────────────────────────────────┘
```

Both judges evaluate the same response independently.
A **PASSED** test proves a real LLM was called — session-level API health
probes run before the first marked test and skip the entire suite if the
API is unreachable (wrong key, no credits, network issue).

---

## What This Tests

| Module | What It Validates |
|---|---|
| `test_accuracy.py` | 10 gold-standard recipes must match exactly; overall accuracy ≥ 90% |
| `test_edge_cases.py` | Model never crashes on empty, None, emoji, long, or unknown inputs |
| `test_prompt_stability.py` | JSON format enforced; tone injections don't corrupt output |
| `test_prompt_injection.py` | 3-layer adversarial suite: content injection, system_prompt override, combined attacks |
| `test_llm_judge.py` | Python rubric scorer + real Claude Haiku judge + real GPT-4o-mini judge + dual-judge consensus |
| `test_retry_flaky.py` | Flaky responses handled with `tenacity` retry (up to 5 attempts) |
| `test_soft_assertions.py` | Partial-correctness scoring with `pytest-check` |
| `test_performance.py` | Response time SLA (< 5 s); benchmark with `pytest-benchmark` |

---

## Project Structure

```
ai-validation-framework/
├── ai_model.py                    # Claude Haiku API + deterministic mock fallback
├── llm_judge.py                   # LLM-as-Judge: Claude Haiku + GPT-4o-mini judges
├── golden_dataset.py              # Single source of truth — 10 gold ingredient→recipe pairs
├── tests/
│   ├── conftest.py                # Fixtures, markers, API health probes, dotenv loader
│   ├── test_accuracy.py           # Gold dataset accuracy tests
│   ├── test_edge_cases.py         # Fuzzing / boundary value tests
│   ├── test_prompt_stability.py   # Prompt engineering validation
│   ├── test_prompt_injection.py   # Security: adversarial prompt injection tests
│   ├── test_llm_judge.py          # Statistical rubric + real LLM-as-Judge tests
│   ├── test_retry_flaky.py        # Retry logic for flaky AI responses
│   ├── test_soft_assertions.py    # Partial correctness scoring
│   └── test_performance.py        # SLA + benchmark tests
├── .github/workflows/
│   └── ai-validation.yml          # GitHub Actions CI/CD (3 independent steps)
├── requirements.txt
├── .env.example                   # Key template — copy to .env for local dev
└── .gitignore
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Gagasr10/ai-validation-framework.git
cd ai-validation-framework
pip install -r requirements.txt

# 2. Run in mock mode (no API key needed — always passes)
pytest tests/ -v --tb=short --ignore=tests/test_performance.py -m "not slow"

# 3. Run real Claude API + judge tests
pytest tests/ -v --tb=short -m "real_api"

# 4. Run GPT-4o-mini judge tests
pytest tests/ -v --tb=short -m "openai_judge"

# 5. Run both judges (dual-judge consensus included)
pytest tests/ -v --tb=short -m "real_api or openai_judge"

# 6. Performance benchmarks
pytest tests/test_performance.py --benchmark-only
```

---

## API Keys

### Local development

Copy `.env.example` to `.env` and fill in your keys:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

`conftest.py` loads `.env` automatically via `python-dotenv`.
The file is gitignored — never committed.

### CI/CD (GitHub Actions)

Add keys as repository secrets: **Settings → Secrets and variables → Actions**

| Secret name | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Claude Haiku SUT + Claude judge |
| `OPENAI_KEY` | GPT-4o-mini judge |

---

## Test Markers

| Marker | Requires | Skipped when |
|---|---|---|
| *(none)* | Nothing | Never — mock always works |
| `@pytest.mark.real_api` | `ANTHROPIC_API_KEY` | Key absent or API unreachable |
| `@pytest.mark.openai_judge` | `OPENAI_KEY` | Key absent or API unreachable |
| Both markers | Both keys | Either key missing |

Tests **skip** (not fail) when infrastructure is unavailable.
Tests **pass** only when the real API was confirmed reachable by the session probe.
Tests **fail** only when the code itself has a bug.

---

## LLM-as-Judge

`llm_judge.py` sends a second independent API call to evaluate the recipe recommendation
on four dimensions:

| Dimension | Weight | Question |
|---|---|---|
| `relevance` | 40% | Does the recipe use the input ingredients as core components? |
| `feasibility` | 20% | Is the cooking time realistic? |
| `confidence_calibration` | 20% | Is the model's confidence score appropriate? |
| `coherence` | 20% | Is the recipe name a recognisable dish? |

```python
from llm_judge import judge_response

# Claude Haiku as judge
verdict = judge_response(["eggs", "flour", "milk"], result)

# GPT-4o-mini as judge (provider auto-detected from model name)
verdict = judge_response(["eggs", "flour", "milk"], result, model="gpt-4o-mini")

# verdict dict
{
    "score": 0.94,
    "verdict": "pass",
    "reasoning": "Pancakes perfectly matches eggs, flour and milk with realistic time.",
    "dimensions": {"relevance": 1.0, "feasibility": 1.0,
                   "confidence_calibration": 0.8, "coherence": 1.0},
    "mode": "real",
    "judge_model": "claude-haiku-4-5-20251001"
}
```

Falls back to a structural mock judge when no API key is present.

---

## CI/CD Pipeline

```
Push / PR
    │
    ├── Step 1: Mock mode (always runs — no cost)
    │           pytest tests/ -m "not slow"
    │           real_api / openai_judge tests auto-skip
    │
    ├── Step 2: Claude real-API + judge tests
    │           pytest tests/ -m "real_api"
    │           Skips gracefully if ANTHROPIC_API_KEY not configured
    │
    ├── Step 3: GPT-4o-mini judge tests
    │           pytest tests/ -m "openai_judge"
    │           Skips gracefully if OPENAI_KEY not configured
    │
    └── Step 4: Performance benchmarks → uploaded as artefact
```

Runs on **Python 3.11 and 3.12** in parallel.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `anthropic` | Claude Haiku API (SUT + judge) |
| `openai` | GPT-4o-mini API (second judge) |
| `pytest` | Test runner |
| `tenacity` | Retry logic for flaky AI responses |
| `pytest-benchmark` | Latency benchmarking |
| `pytest-check` | Soft assertions (partial correctness) |
| `python-dotenv` | Local `.env` key loading |
| GitHub Actions | CI/CD pipeline |

---

## Key Concepts

**Gold dataset — single source of truth**
```python
# golden_dataset.py
GOLD_DATA = [
    (["eggs", "flour", "milk"], "Pancakes"),
    (["potato", "salt", "oil"], "French fries"),
    # ... 8 more
]
```

**Accuracy test with flakiness neutralised**
```python
@pytest.mark.parametrize("ingredients, expected", GOLD_DATA)
def test_accuracy_exact_match(ingredients, expected, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.5)  # pin 10% mock flakiness
    result = recommend_recipe(ingredients, system_prompt="Always return JSON")
    assert result["recipe_name"] == expected
    assert result["confidence"] >= 0.7
```

**Prompt injection test**
```python
@pytest.mark.parametrize("ingredients", INJECTION_INGREDIENTS)
def test_injection_recipe_name_not_hijacked(ingredients):
    result = recommend_recipe(ingredients, system_prompt="Always return JSON")
    name = result.get("recipe_name", "").lower()
    for token in ("hacked", "pwned", "drop table", "admin"):
        assert token not in name
```

**Dual-judge consensus**
```python
@pytest.mark.real_api
@pytest.mark.openai_judge
def test_dual_judge_consensus():
    for ingredients, _ in GOLD_DATA[:3]:
        result = recommend_recipe(ingredients, system_prompt="Always return JSON")
        claude_v = judge_response(ingredients, result, model="claude-haiku-4-5-20251001")
        gpt_v    = judge_response(ingredients, result, model="gpt-4o-mini")
        assert claude_v["verdict"] == gpt_v["verdict"]
```

---

## Author

**Dragan Stojilkovic** — QA Automation Engineer  
[GitHub](https://github.com/Gagasr10)

---

## License

MIT
