# Contributing to Open WebUI AB-MCTS & Multi-Model Pipeline Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Open WebUI AB-MCTS & Multi-Model Pipeline Project.

## 🚀 Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Git
- Ollama with models: `deepseek-r1:1.5b`, `gemma3:1b`, `llama3.2:1b`

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/openwebui-ab-mcts.git`
3. Initialize submodules: `git submodule update --init --recursive`
4. Start services: `docker-compose up -d`
5. Access Open WebUI at `http://localhost:3000` and Backend API docs at `http://localhost:8095/api/docs`

## 📝 Contribution Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused

### Commit Messages
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests

### Pull Request Process
1. Create a feature branch from `main`
2. Make your changes
3. Add tests if applicable
4. Update documentation
5. Submit a pull request

## 🧪 Testing

### Running Tests
```bash
# Run all tests
docker-compose exec backend-api python -m pytest

# Run specific test file
docker-compose exec backend-api python -m pytest tests/test_ab_mcts.py
```

### Test Coverage
- Aim for >80% test coverage
- Write unit tests for new features
- Include integration tests for API endpoints

## 📚 Documentation

### Updating Documentation
- Update README.md for major changes
- Update API_REFERENCE.md for API changes
- Update ARCHITECTURE.md for architectural changes
- Update DEPLOYMENT.md when docker-compose or commands change
- Update PROJECT_STATUS.md and PROJECT_HANDOFF_PROMPT.md when limitations/status change
- Add inline comments for complex logic

## 🐛 Bug Reports

When reporting bugs, please include:
- Description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Docker version, etc.)

## 💡 Feature Requests

When requesting features, please include:
- Description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Alternative solutions considered

## 📞 Getting Help

- Create an issue for questions
- Join our discussions for general chat
- Check existing issues before creating new ones

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing! 🎉
