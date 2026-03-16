# Contributing to Dejavu

Thanks for your interest in contributing! Here's how to get started.

## Development setup

```bash
# Clone the repo
git clone https://github.com/peterhollens/dejavu.git
cd dejavu

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[vec,daemon,fallback]"

# Install test dependencies
pip install pytest pytest-asyncio
```

## Running tests

```bash
pytest
```

## Project structure

- `src/dejavu/` -- all source code
- `tests/` -- test suite
- `pyproject.toml` -- project metadata and dependencies

## Guidelines

- Keep changes focused -- one feature or fix per PR
- Add tests for new functionality
- Follow existing code style (type hints, docstrings on public APIs)
- Run tests before submitting

## Reporting issues

Please include:
- Python version
- OS
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
