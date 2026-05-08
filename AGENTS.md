# Repository Guidelines

## Project Structure & Module Organization
This repository is a small evaluation harness for a single submission file.
- `solution.py`: the only file you should modify for model logic. Implement `MyHarness` here.
- `harness_base.py`: fixed base class and interface contract.
- `run.py`: local evaluation script using `data/train_dev.jsonl` and `data/test_dev.jsonl`.
- `llm_client.py`: OpenAI-compatible client configuration for local runs.
- `data/`: dev split JSONL files used by the runner.
- `tokenizer/`: bundled tokenizer assets used for token counting.

## Build, Test, and Development Commands
Use Python 3 with the dependencies in `requirements.txt`.
- `pip install -r requirements.txt`: install runtime dependencies.
- `python run.py`: run the local benchmark against the dev split.
- `python run.py --workers 100`: increase prediction concurrency when LLM latency is high.
- `python llm_client.py`: quick connectivity check for the configured endpoint.

## Coding Style & Naming Conventions
Follow the existing Python style in the repo: 4-space indentation, type hints where already used, and short, direct helper functions. Keep imports limited to the standard library, `numpy`, and `harness_base` inside `solution.py`; do not add other third-party dependencies there. Use `MyHarness` for the submission class and keep helper method names descriptive and lowercase with underscores.

## Testing Guidelines
There is no formal test suite. Validate changes by running `python run.py` and checking both accuracy and error output. The runner reports prompt/completion token usage and surfaces exceptions per sample, so treat regressions in those numbers as a signal that prompt handling changed.

## Commit & Pull Request Guidelines
This checkout does not include Git history, so no repository-specific commit convention is available here. Use concise, imperative commit subjects such as `fix harness prompt truncation`. For pull requests, include a short summary of the behavior change, the validation command you ran, and any output or screenshots only if they help explain a regression.

## Agent-Specific Instructions
Do not edit `harness_base.py`, `run.py`, or the tokenizer assets unless the task explicitly requires it. Avoid reading or writing files from inside `solution.py`; the harness rules forbid disk access during scoring.
