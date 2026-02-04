# Contributing to BathySim

This library implements a high-fidelity Topobathymetric Fusion Engine.

## Core Principles

1. **Type Safety**: All new code must be fully type-hinted and pass `mypy` strict checks.
2. **Test Coverage**: Maintain >90% test coverage using `pytest`.
3. **Generalization**: Avoid hardcoding specific biologger simulation logic; keep components reusable.
4. **Standardized Interfaces**: Implement BMI standards where applicable to ensure interoperability.

## Development Setup

We recommend using **Micromamba** (or Mamba/Conda) for development, as it handles complex binary dependencies like `PDAL` and `GDAL` much more reliably than pip.

### Option A: Micromamba (Recommended)

1. **Install Micromamba**: Follow instructions at [mamba.readthedocs.io](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
2. **Create Environment**:

    ```bash
    micromamba create -f environment.yml
    micromamba activate topobathysim
    ```

3. **Install Package in Editable Mode**:

    ```bash
    pip install -e .
    ```

### Option B: Python Venv (Pip)

If you prefer standard python tools, ensuring you have system-level libraries for `gdal` and `pdal` installed first.

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with test dependencies
pip install -e ".[test]"
```

## Development Workflow

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to ensure code quality (linting, formatting, type checking) before items are committed.

1. **Install pre-commit**:

    ```bash
    pip install pre-commit
    ```

2. **Install the hooks**:

    ```bash
    pre-commit install
    ```

3. **Run manually** (optional, recommended before commit):

    ```bash
    pre-commit run --all-files
    ```

Hooks configured:

- `ruff`: For linting and formatting (replaces flake8/isort/black).
- `mypy`: Static type checking.
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`.

## Running Tests

```bash
pytest
```

## Submitting Changes

1. Fork the repository.
2. Create a feature branch.
3. Add tests for your feature.
4. Update `README.rst` if necessary.
5. Create a Pull Request (PR) describing the changes.
