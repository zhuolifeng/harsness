# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a banking support message classification harness. The task is to classify customer messages (e.g., "My card is stuck in an ATM") into intent labels (e.g., `card_swallowed`). The system uses a hybrid approach: TF-IDF/Jaccard lexical ranking combined with LLM-based classification for ambiguous cases.

## Commands

- **Install dependencies:** `pip install -r requirements.txt`
- **Run local evaluation:** `python run.py`
- **Run with higher concurrency:** `python run.py --workers 100`
- **Test LLM connectivity:** `python llm_client.py`

`run.py` runs multiple evaluation rounds (default 4), reports per-round accuracy, average accuracy, token usage per sample, and any prediction errors.

## Architecture

```
harness_base.py    Harness base class (DO NOT MODIFY) — defines update()/predict() interface
solution.py        MyHarness implementation (ONLY file to modify for model logic)
llm_client.py      OpenAI-compatible LLM client config — edit BASE_URL/API_KEY/MODEL for your endpoint
run.py             Local evaluation runner with concurrency, token tracking, prompt truncation
data/              JSONL files with {"text", "label"} records (train_dev.jsonl, test_dev.jsonl)
tokenizer/         Bundled tokenizer assets for token counting (used by llm_client.count_tokens)
```

### MyHarness Flow (solution.py)

1. **`update(text, label)`** — Ingests labeled training examples. Builds per-label inverted index with word tokens, character n-grams (3,4), and IDF statistics.
2. **`predict(text)`** — Two-stage classification:
   - **Stage 1 (lexical ranking):** Ranks all labels by weighted combination of TF-IDF cosine similarity (0.57), character n-gram Jaccard (0.28), and label-token Jaccard (0.15), plus a phrase-containment bonus. If the top label has score ≥ 0.55 with margin ≥ 0.12, returns immediately without LLM.
   - **Stage 2 (LLM):** Builds a prompt with top-ranked candidate labels, their keywords and example texts, then asks the LLM to pick one. Falls back to the lexical top label if LLM fails or is unavailable.

### Key Constraints

- `solution.py` may only import stdlib, `numpy`, and `harness_base` — no other third-party libraries.
- No disk I/O is allowed inside `solution.py` during scoring.
- Prompts exceeding `max_prompt_tokens` (default 2048) are automatically truncated by the runner.
- The dev split is for local testing only; final scoring uses different train/test data — do not overfit.

## Coding Conventions

- Python 3, 4-space indentation, type hints where already used.
- Helper methods in `MyHarness` use leading underscores (`_tokens`, `_rank_labels`, etc.).
- Imports inside `MyHarness` are aliased with underscores (`_math`, `_re`, `_Counter`).
- Keep `solution.py` self-contained; do not add external dependencies.

## Files You Should Not Modify

`harness_base.py`, `run.py`, `llm_client.py` (structure), tokenizer assets, and data files — unless the task explicitly requires it.
