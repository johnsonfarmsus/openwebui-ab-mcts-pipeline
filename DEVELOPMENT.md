# Development Guide

## 🛠️ Development Setup

### Prerequisites

- **Python**: 3.11 or higher
- **Node.js**: 18 or higher (for frontend development)
- **Docker**: For containerized development
- **Git**: Version control

### Local Development

#### 1. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd openwebui-setup

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### 2. Development Dependencies

```bash
# Install development tools
pip install pytest black flake8 mypy pre-commit

# Install pre-commit hooks
pre-commit install
```

#### 3. Environment Configuration

```bash
# Copy development environment
cp .env.example .env.dev

# Edit development settings
nano .env.dev
```

**Development Environment:**
```bash
# Development settings
LOG_LEVEL=debug
DEBUG=true
ENABLE_HOT_RELOAD=true

# Local Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Development ports
AB_MCTS_PORT=8094
MULTI_MODEL_PORT=8090
BACKEND_PORT=8095
```

## 🏗️ Project Structure

```
openwebui-setup/
├── README.md                    # Project overview
├── ARCHITECTURE.md              # System architecture
├── API_REFERENCE.md             # API documentation
├── DEPLOYMENT.md                # Deployment guide
├── DEVELOPMENT.md               # This file
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── docker-compose.yml           # Development orchestration
├── docker-compose.prod.yml      # (Optional) production orchestration
├── Dockerfile                   # Container definition
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── .pre-commit-config.yaml      # Pre-commit hooks
├── pytest.ini                  # Test configuration
├── pyproject.toml              # Python project config
├── pipelines/                   # (Legacy/optional) pipeline artifacts
│   └── ab_mcts_pipeline.py
├── backend/                     # Backend & services
│   ├── __init__.py
│   ├── api/                    # FastAPI endpoints
│   │   ├── __init__.py
│   │   ├── main.py            # API main (8095)
│   │   ├── models.py          # Model management endpoints
│   │   ├── pipelines.py       # Pipeline endpoints
│   │   └── monitoring.py      # Monitoring endpoints
│   ├── services/              # Business logic
│   │   ├── __init__.py
│   │   ├── proper_treequest_ab_mcts_service.py # AB‑MCTS service (8094)
│   │   ├── proper_multi_model_service.py       # Multi‑Model service (8090)
│   │   ├── experiment_logger.py               # Runs (SQLite + JSONL)
│   │   └── config_manager.py                  # Config management
│   ├── model_integration.py    # OpenAI‑compatible models (8098)
│   └── openwebui_integration.py# Tools (8097)
│   ├── models/                # Data models
│   │   ├── __init__.py
│   │   ├── llm_state.py       # LLM state model
│   │   ├── search_stats.py    # Search statistics
│   │   └── model_config.py    # Model configuration
│   └── dashboard/             # Web dashboard (static)
├── services/                  # Standalone services
│   ├── ab_mcts_service.py    # AB-MCTS service
│   ├── multi_model_service.py # Multi-model service
│   └── treequest_ab_mcts_service.py # TreeQuest implementation
├── interfaces/                # User interfaces
│   ├── dashboard.html
│   ├── conversational_ab_mcts_interface.html
│   └── tool_test.html
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Test configuration
│   ├── test_ab_mcts.py       # AB-MCTS tests
│   ├── test_multi_model.py   # Multi-model tests
│   ├── test_backend.py       # Backend tests
│   └── test_integration.py   # Integration tests
├── docs/                      # Documentation
│   ├── api/                  # API documentation
│   ├── architecture/         # Architecture docs
│   └── user_guide/           # User guides
└── scripts/                   # Utility scripts
    ├── setup_dev.sh          # Development setup
    ├── run_tests.sh          # Test runner
    └── deploy.sh             # Deployment script
```

## 🔧 Development Workflow

### 1. Code Style

We use **Black** for code formatting and **Flake8** for linting.

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

### 2. Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ab_mcts.py

# Run with coverage
pytest --cov=.

# Run integration tests
pytest tests/test_integration.py
```

### 3. Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 🧪 Testing

### Test Structure

```python
# tests/test_ab_mcts.py
import pytest
from backend.services.proper_treequest_ab_mcts_service import ProperTreeQuestABMCTSService

class TestABMCTS:
    def test_basic_query(self):
        service = ProperTreeQuestABMCTSService()
        result = service.process_query("What is AI?")
        assert result.success == True
        assert len(result.result) > 0
    
    def test_anti_hallucination(self):
        service = ProperTreeQuestABMCTSService()
        result = service.process_query("How many chocolate chips?")
        assert "I don't have verified information" in result.result
```

### Mocking

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_ollama():
    with patch('backend.services.proper_treequest_ab_mcts_service.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response"}
        mock_post.return_value = mock_response
        yield mock_post
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
import requests

def test_ab_mcts_api():
    response = requests.post(
        "http://localhost:8094/query",
        json={"query": "What is machine learning?", "iterations": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "search_stats" in data
```

## 🔄 Development Services

### Running Services Locally

```bash
# Start Ollama
ollama serve

# Start AB-MCTS service
python backend/services/proper_treequest_ab_mcts_service.py

# Start Multi-Model service
python backend/services/proper_multi_model_service.py

# Start Backend API
python backend/api/main.py
```

### Docker Development

```bash
# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f

# Rebuild service
docker-compose build ab-mcts-service
docker-compose up -d ab-mcts-service
```

## 📝 Adding New Features

### 1. Feature Branch

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes
# ... code changes ...

# Commit changes
git add .
git commit -m "Add new feature"

# Push branch
git push origin feature/new-feature
```

### 2. Code Review

```bash
# Create pull request
# ... GitHub PR process ...

# Address review comments
# ... make changes ...

# Update PR
git add .
git commit -m "Address review comments"
git push origin feature/new-feature
```

### 3. Testing

```bash
# Run tests
pytest

# Run specific tests
pytest tests/test_new_feature.py

# Check coverage
pytest --cov=. --cov-report=html
```

## 🐛 Debugging

### 1. Local Debugging

```python
# Add debug prints
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger
import pdb
pdb.set_trace()

# Logging
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### 2. Docker Debugging

```bash
# Run container in debug mode
docker run -it --rm openwebui-setup-ab-mcts-service python -m pdb treequest_ab_mcts_service.py

# Check container logs
docker logs -f container-name

# Execute commands in container
docker exec -it container-name /bin/bash
```

### 3. Performance Debugging

```python
# Profile code
import cProfile
cProfile.run('your_function()')

# Memory profiling
from memory_profiler import profile

@profile
def your_function():
    # ... code ...
    pass
```

## 📊 Monitoring Development

### 1. Local Monitoring

```bash
# Check service health
curl http://localhost:8094/health

# Monitor performance
curl http://localhost:8095/api/monitoring/performance

# View logs
tail -f logs/ab_mcts.log
```

### 2. Development Dashboard

```bash
# Access static dashboard (served by docker-compose http-server)
open http://localhost:8081/dashboard.html
```

## 🔧 Configuration Management

### 1. Environment Variables

```python
# config.py
import os
from typing import Optional

class Config:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "50"))
    MAX_DEPTH: int = int(os.getenv("MAX_DEPTH", "10"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        return all([
            cls.OLLAMA_BASE_URL,
            cls.LOG_LEVEL in ["debug", "info", "warning", "error"]
        ])
```

### 2. Model Configuration

```python
# models/model_config.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    id: str
    name: str
    endpoint: str
    parameters: Dict[str, Any]
    enabled: bool = True
    performance_score: float = 0.0
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "endpoint": self.endpoint,
            "parameters": self.parameters,
            "enabled": self.enabled,
            "performance_score": self.performance_score,
            "usage_count": self.usage_count
        }
```

## 🚀 Deployment

### 1. Local Deployment

```bash
# Build and start services
docker-compose up -d --build

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 2. Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
curl http://production-url/health
```

## 📚 Documentation

### 1. API Documentation

```python
# Generate API docs
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="AB-MCTS API",
        version="1.0.0",
        description="API for AB-MCTS and Multi-Model services",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

### 2. Code Documentation

```python
def process_query(self, query: str, iterations: int = 20) -> QueryResponse:
    """
    Process a query using AB-MCTS algorithm.
    
    Args:
        query: The user's question
        iterations: Number of search iterations
        
    Returns:
        QueryResponse with result and statistics
        
    Raises:
        ValueError: If query is empty
        ModelError: If model call fails
    """
    pass
```

## 🤝 Contributing

### 1. Code Standards

- **Python**: Follow PEP 8
- **Type Hints**: Use type hints for all functions
- **Docstrings**: Document all public functions
- **Tests**: Write tests for new features
- **Commits**: Use conventional commit messages

### 2. Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

### 3. Code Review

- **Review Checklist**: Check code quality, tests, documentation
- **Testing**: Verify all tests pass
- **Performance**: Check for performance implications
- **Security**: Review for security issues

## 📞 Getting Help

### 1. Documentation

- **README.md**: Project overview
- **ARCHITECTURE.md**: System design
- **API_REFERENCE.md**: API documentation
- **DEPLOYMENT.md**: Deployment guide

### 2. Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Discord**: Real-time chat and support

### 3. Development Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **TreeQuest Docs**: https://github.com/SakanaAI/treequest
- **Open WebUI Docs**: https://github.com/open-webui/open-webui
