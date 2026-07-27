# Contributing

## Local setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install `.[topic]`, `.[lstm]`, or `.[all]` when changing optional machine
learning paths.

## Development checks

Run the same checks used by CI:

```bash
python -m ruff check .
python -m pytest
python -m compileall -q src
```

## Pull requests

- Keep one coherent concern per pull request.
- Add or update tests for behavior changes.
- Do not commit raw licensed news, private datasets, model checkpoints, or
  generated artifacts.
- Describe the chronological split and leakage implications of modelling
  changes.
- Report exact commands used for validation.

## Commit style

Use a short conventional prefix when it clarifies intent:

- `feat:` user-visible functionality
- `fix:` defect correction
- `test:` test-only changes
- `docs:` documentation
- `build:` packaging and dependency changes
- `ci:` workflow changes
