# 🤖 AI Model Validation Framework

> A production-ready Python + pytest framework for validating AI model outputs —
> accuracy, performance, edge cases, prompt stability, and flakiness handling.

Supports **real Claude API** (Anthropic) out of the box.  
Falls back to a **deterministic mock** automatically when no API key is set —
so CI/CD runs with zero cost and zero flakiness.

[![AI Model Validation](https://github.com/Gagasr10/ai-validation-framework/actions/workflows/ai-validation.yml/badge.svg)](https://github.com/Gagasr10/ai-validation-framework/actions)

---

## 🧪 What This Tests

| Module | What It Validates |
|--------|-------------------|
| `test_accuracy.py` | 10 gold-standard recipes must match exactly; overall accuracy ≥ 90% |
| `test_performance.py` | Response time SLA (< 5 s); benchmark with `pytest-benchmark` |
| `test_edge_cases.py` | Model never crashes on empty, None, emoji, long, or unknown inputs |
| `test_prompt_stability.py` | JSON format enforced; tone injections don't corrupt output |
| `test_retry_flaky.py` | Flaky responses handled with `tenacity` retry (up to 5 attempts) |
| `test_soft_assertions.py` | Partial-correctness scoring with `pytest-check` |

---

## 🏗️ Project Structure

```
ai-validation-framework/
├── ai_model.py                   # Claude API + mock fallback
├── tests/
│   ├── conftest.py               # Shared fixtures and hooks
│   ├── test_accuracy.py          # Gold dataset accuracy tests
│   ├── test_performance.py       # SLA + benchmark tests
│   ├── test_edge_cases.py        # Fuzzing / boundary values
│   ├── test_prompt_stability.py  # Prompt engineering validation
│   ├── test_retry_flaky.py       # Retry logic for flaky AI
│   └── test_soft_assertions.py   # Partial correctness scoring
├── .github/workflows/
│   └── ai-validation.yml         # GitHub Actions CI/CD
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Gagasr10/ai-validation-framework.git
cd ai-validation-framework
pip install -r requirements.txt

# 2. Run in MOCK mode (no API key needed)
pytest tests/ -v --tb=short

# 3. Performance benchmarks only
pytest tests/test_performance.py --benchmark-only

# 4. Smoke tests only
pytest tests/ -m smoke -v
```

> **No API key required.** The mock runs locally, simulates realistic AI behaviour
> (random delays, occasional wrong answers), and costs nothing.

---

## 🔑 Real API vs Mock Mode

| Mode | When | Behaviour |
|------|------|-----------|
| **Real API** | `ANTHROPIC_API_KEY` is set | Calls Claude Haiku, parses JSON response |
| **Mock** | No API key (default) | Deterministic local simulation, 10% intentional flakiness |

The mock is **not a test double for convenience** — it deliberately simulates
real-world AI behaviour: random delays, occasional wrong answers, edge case crashes.

The 10% flakiness exists specifically to exercise `test_retry_flaky.py`.
Accuracy tests use `monkeypatch` to neutralise flakiness so they always pass deterministically.

---

## ⚙️ CI/CD — GitHub Actions

The workflow runs automatically on every push and pull request:

1. **Mock mode** — always runs, zero cost, tests the framework itself
2. Uploads benchmark results as artefacts
3. Runs on both Python 3.11 and 3.12

```yaml
# .github/workflows/ai-validation.yml
pytest tests/ -v --tb=short
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pytest` | Test runner |
| `anthropic` | Claude API client |
| `tenacity` | Retry logic for flaky AI responses |
| `pytest-benchmark` | Latency benchmarking |
| `pytest-check` | Soft assertions (partial correctness) |
| `python-dotenv` | Environment variable management |
| GitHub Actions | CI/CD pipeline |

---

## 💡 Key Concepts Demonstrated

**Accuracy Testing with Gold Dataset**
```python
GOLD_DATA = [
    (["eggs", "flour", "milk"], "Pancakes"),
    # ... 9 more known recipes
]
@pytest.mark.parametrize("ingredients, expected", GOLD_DATA)
def test_accuracy_exact_match(ingredients, expected, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.5)  # neutralise flakiness
    result = recommend_recipe(ingredients, system_prompt="Always return JSON")
    assert result["recipe_name"] == expected
```

**Retry for Flaky AI**
```python
@retry(stop=stop_after_attempt(5), wait=wait_fixed(0.2), reraise=True)
def stable_call(ingredients):
    result = recommend_recipe(ingredients, system_prompt="Always return JSON")
    assert result["recipe_name"] == "Pancakes"
    return result
```

**Soft Assertions (partial correctness)**
```python
def test_partial_correctness(result):
    check.equal(result["recipe_name"], "Pancakes")
    check.between(result["time_min"], 5, 30)
    check.greater(result["confidence"], 0.7)
    # All checks run even if one fails
```

---

## 👤 Author

**Dragan Stojilkovic** — QA Automation Engineer  
[GitHub](https://github.com/Gagasr10) 

---

## 📄 License

MIT
