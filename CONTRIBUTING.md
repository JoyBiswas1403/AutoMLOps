# Contributing to AutoMLOps

Thank you for your interest in contributing to AutoMLOps! This document provides guidelines and instructions for contributing.

## 🚀 Quick Start

### Development Setup

```bash
# Clone the repository
git clone https://github.com/JoyBiswas1403/AutoMLOps.git
cd AutoMLOps

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest -v

# With coverage
pytest --cov=training --cov=serving --cov-report=term-missing
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check . --fix

# Type check
mypy training serving pipelines drift
```

## 📋 Contribution Guidelines

### Code Style

- **Python 3.10+** required
- **Type hints** on all public functions
- **Google-style docstrings** for documentation
- **Ruff** for formatting and linting
- **Mypy** for type checking

### Commit Messages

Use conventional commits:
```
feat: add new feature
fix: fix a bug
docs: update documentation
test: add tests
refactor: code refactoring
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run tests: `pytest`
5. Run linting: `ruff check .`
6. Commit with conventional message
7. Push and create PR

### PR Checklist

- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Type hints added
- [ ] Docstrings added
- [ ] No linting errors
- [ ] Documentation updated

## 🏗️ Project Structure

```
training/src/     # ML training modules
serving/api/      # FastAPI application
pipelines/        # Orchestration scripts
drift/            # Drift detection
tests/unit/       # Unit tests
```

## 🐛 Reporting Issues

When reporting issues, please include:

1. Clear description of the problem
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details (OS, Python version)
5. Relevant logs or error messages

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Questions? Open an issue or reach out to [@JoyBiswas1403](https://github.com/JoyBiswas1403).
