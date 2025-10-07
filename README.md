# AB-MCTS & Multi-Model Reasoning Engine for Open WebUI

Advanced reasoning models for Open WebUI using Adaptive Branching Monte Carlo Tree Search (AB-MCTS) and Multi-Model collaboration.

## 🎯 Project Overview

This project implements **Sakana AI's AB-MCTS (Adaptive Branching Monte Carlo Tree Search)** algorithm and a **Multi-Model** collaboration system, both integrated with Open WebUI as selectable AI models for advanced reasoning and decision-making.

### Key Features
- **AB-MCTS Pipeline**: Advanced tree search with quality scoring and anti-hallucination
- **Multi-Model Pipeline**: Multi-model collaboration for comprehensive answers
- **OpenAI-Compatible API**: Native integration with Open WebUI's model system
- **Real-time Monitoring**: Prometheus metrics and Grafana dashboards
- **Experiment Logging**: SQLite + JSONL run tracking for research and analysis

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Open WebUI Interface                     │
├─────────────────────────────────────────────────────────────┤
│  Model Selection:                                           │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   ab-mcts       │  │  multi-model    │                  │
│  │                 │  │                 │                  │
│  │ • Tree Search   │  │ • Collaboration │                  │
│  │ • Deep Analysis │  │ • Multi-perspective                │
│  │ • Best Quality  │  │ • Comprehensive │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Integration Service (8098)               │
│                  OpenAI-Compatible API                      │
└─────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│   AB-MCTS Service       │    │  Multi-Model Service    │
│   (port 8094)           │    │  (port 8090)            │
│                         │    │                         │
│ • TreeQuest Algorithm   │    │ • Direct Collaboration  │
│ • Thompson Sampling     │    │ • Model Voting         │
│ • Anti-Hallucination    │    │ • Synthesis            │
└─────────────────────────┘    └─────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                        Ollama                               │
│              Local LLM Inference Engine                     │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
openwebui-setup/
├── README.md                           # This file
├── docker-compose.yml                  # Docker orchestration
├── Dockerfile                          # Container definition
├── requirements.txt                    # Python dependencies
├── backend/
│   ├── api/
│   │   └── main.py                     # Management API (port 8095)
│   ├── services/
│   │   ├── proper_treequest_ab_mcts_service.py  # AB-MCTS (port 8094)
│   │   ├── proper_multi_model_service.py        # Multi-Model (port 8090)
│   │   ├── experiment_logger.py                 # Run logging
│   │   └── config_manager.py                    # Configuration
│   ├── model_integration.py           # OpenAI-compatible model API (8098)
│   └── openwebui_integration.py       # Tool endpoints (8097)
├── interfaces/
│   ├── dashboard.html                  # Management dashboard
│   └── idiots_guide.html              # Setup guide
└── logs/                               # Experiment logs and runs
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Ollama running locally (port 11434)
- Recommended models: `llama3.2:latest`, `qwen2.5:latest`, `deepseek-r1:1.5b`

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/openwebui-setup.git
cd openwebui-setup
```

2. **Pull Ollama models**:
```bash
ollama pull llama3.2:latest
ollama pull qwen2.5:latest
ollama pull deepseek-r1:1.5b
```

3. **Start all services**:
```bash
docker-compose up -d
```

4. **Verify services**:
```bash
docker-compose ps
```

All services should show "Up" status.

### Connecting to Open WebUI

1. Open Open WebUI at `http://localhost:3000`

2. **Add the model provider**:
   - Click your profile → **Settings**
   - Go to **Connections**
   - Click **+ Add Connection**
   - Select **OpenAI**
   - **API Base URL**: `http://model-integration:8098`
   - **API Key**: `dummy-key` (any value works)
   - Click **Verify Connection** → Should show "✓ Connected"
   - Click **Save**

3. **Select a model**:
   - Start a new chat
   - Click the model dropdown
   - Select either:
     - **ab-mcts** - Advanced tree search reasoning
     - **multi-model** - Collaborative AI

### Using the Models

#### AB-MCTS
**Best for:**
- Complex problem solving
- Multi-step reasoning
- Strategic planning
- Mathematical proofs
- Decision trees

**Example queries:**
- "Design a distributed caching system for a social media platform"
- "Prove that the square root of 2 is irrational"
- "What's the optimal strategy for a two-player game where..."

**Note:** Responses may take 30-120 seconds due to tree search exploration.

#### Multi-Model
**Best for:**
- Comprehensive analysis
- Multiple perspectives
- Research questions
- Balanced viewpoints
- Faster responses

**Example queries:**
- "Compare microservices vs monolithic architectures"
- "Analyze the pros and cons of remote work"
- "Explain quantum computing to different audiences"

## 🔧 Services

| Service | Port | Description |
|---------|------|-------------|
| Open WebUI | 3000 | Main chat interface |
| Model Integration | 8098 | OpenAI-compatible model API |
| AB-MCTS Service | 8094 | TreeQuest AB-MCTS implementation |
| Multi-Model Service | 8090 | Multi-model collaboration |
| Backend API | 8095 | Management dashboard API |
| MCP Server | 8096 | Model Context Protocol bridge |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3001 | Dashboards and visualization |
| HTTP Server | 8081 | Static interfaces |

## ⚙️ Configuration

### AB-MCTS Parameters

Configure via the Backend API:

```bash
curl -X POST http://localhost:8095/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "ab_mcts_iterations": 20,
    "ab_mcts_max_depth": 5
  }'
```

**Parameters:**
- `ab_mcts_iterations`: Number of search iterations (1-100, default: 20)
  - Higher = better quality, slower response
  - Recommended: 10-20 for most queries
- `ab_mcts_max_depth`: Maximum tree depth (1-20, default: 5)
  - Higher = deeper reasoning, slower response
  - Recommended: 3-5 for most queries

### Model Selection

Update which Ollama models each service uses:

```bash
# AB-MCTS models
curl -X POST http://localhost:8094/models/update \
  -H "Content-Type: application/json" \
  -d '{"models": ["llama3.2:latest", "qwen2.5:latest"]}'

# Multi-Model models
curl -X POST http://localhost:8090/models/update \
  -H "Content-Type: application/json" \
  -d '{"models": ["llama3.2:latest", "qwen2.5:latest", "deepseek-r1:1.5b"]}'
```

## 📊 Monitoring

### Prometheus Metrics

Access Prometheus at `http://localhost:9090`

**Key metrics:**
- `model_integration_requests_total` - Total requests by model
- `model_integration_success_total` - Successful responses
- `model_integration_failures_total` - Failed responses
- `model_integration_latency_seconds` - Response time histogram
- `model_integration_active_queries` - Current active queries

### Grafana Dashboards

Access Grafana at `http://localhost:3001` (credentials: `admin/admin`)

**Pre-configured dashboards:**
- Request rates and success rates
- Latency percentiles (p50, p95, p99)
- Active query monitoring
- Error rates by type
- Service health status

### Experiment Logs

All runs are logged to `/app/logs`:
- `logs/runs.db` - SQLite index
- `logs/runs/YYYYMMDD/run_<id>.jsonl` - Event stream per run

**View logs:**
- Dashboard: `http://localhost:8081/dashboard.html` → "Runs" tab
- API: `GET http://localhost:8095/api/runs?limit=50`
- Details: `GET http://localhost:8095/api/runs/{run_id}`

## 🐛 Troubleshooting

### Models not appearing in Open WebUI

**Check model integration service:**
```bash
curl http://localhost:8098/health
curl http://localhost:8098/v1/models
```

**Verify Open WebUI connection:**
- Settings → Connections → Verify the connection shows "✓ Connected"
- Try refreshing the page
- Check browser console for errors

### Slow responses

**Reduce AB-MCTS iterations:**
```bash
curl -X POST http://localhost:8095/api/config \
  -d '{"ab_mcts_iterations": 10, "ab_mcts_max_depth": 3}'
```

**Use faster Ollama models:**
```bash
ollama pull llama3.2:1b  # Smaller, faster model
```

**Check Ollama performance:**
```bash
time curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.2:latest","prompt":"test","stream":false}'
```

### Service connection errors

**Check all services are running:**
```bash
docker-compose ps
```

**View service logs:**
```bash
docker logs model-integration
docker logs ab-mcts-service
docker logs multi-model-service
```

**Restart services:**
```bash
docker-compose restart
```

## 🚧 Known Issues

- **Timeouts**: AB-MCTS can take 30-120s on complex queries (streaming keeps UI responsive)
- **Verbosity**: AB-MCTS responses can be lengthy (working on length controls)
- **Quality drift**: Occasional hallucinations (add stricter validation)

## 📚 API Reference

### Model Integration Service (port 8098)

**OpenAI-Compatible Endpoints:**
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions

**Management Endpoints:**
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /performance` - Performance statistics
- `GET /config` - Current configuration
- `POST /config` - Update configuration

### AB-MCTS Service (port 8094)

- `POST /query` - Run AB-MCTS query
  - Body: `{"query": "...", "iterations": 20, "max_depth": 5}`
- `GET /models` - List available models
- `POST /models/update` - Update model selection
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

### Multi-Model Service (port 8090)

- `POST /query` - Run multi-model query
  - Body: `{"query": "..."}`
- `GET /models` - List available models
- `POST /models/update` - Update model selection
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## 🤝 Related Projects

- [Scientific Data Enrichment Tool](https://github.com/yourusername/scientific-enrichment-tool) - Chemistry and materials science enrichment for Open WebUI (separate tool)

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- [Sakana AI](https://sakana.ai/) for AB-MCTS research and TreeQuest
- [Open WebUI](https://github.com/open-webui/open-webui) for the chat interface
- [Ollama](https://ollama.ai/) for local LLM inference
- [Prometheus](https://prometheus.io/) & [Grafana](https://grafana.com/) for observability

## 📖 Additional Documentation

- `ARCHITECTURE.md` - Detailed architecture and design
- `API_REFERENCE.md` - Complete API documentation
- `DEPLOYMENT.md` - Production deployment guide
- `docs/research/RESEARCH_GUIDE.md` - Research and analysis guide
