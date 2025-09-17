# Open WebUI AB-MCTS & Multi-Model Pipeline Project

## 🎯 Project Overview

This project implements **Sakana AI's AB-MCTS (Adaptive Branching Monte Carlo Tree Search)** algorithm and a **Simple Multi-Model** collaboration system, both integrated with Open WebUI for advanced AI reasoning and decision-making.

### Key Features
- **AB-MCTS Pipeline**: Advanced tree search with quality scoring and anti-hallucination
- **Multi-Model Pipeline**: Simple multi-model collaboration for fast responses
- **Unified Backend**: Management dashboard for both pipelines
- **Open WebUI Integration**: Native chat interface with model selection
- **Real-time Monitoring**: Performance analytics and search tree visualization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Open WebUI Interface                     │
├─────────────────────────────────────────────────────────────┤
│  Chat Model Selection:                                      │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   AB-MCTS       │  │  Multi-Model    │                  │
│  │   (Advanced)    │  │  (Simple)       │                  │
│  │                 │  │                 │                  │
│  │ • Tree Search   │  │ • Direct Collab │                  │
│  │ • Deep Analysis │  │ • Fast Response │                  │
│  │ • Best Quality  │  │ • Easy to Use   │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Management                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  AB-MCTS        │  │  Multi-Model    │                  │
│  │  Pipeline       │  │  Pipeline       │                  │
│  │                 │  │                 │                  │
│  │ • TreeQuest     │  │ • Direct API    │                  │
│  │ • Anti-Halluc.  │  │ • Model Voting  │                  │
│  │ • Quality Score │  │ • Fast Synthesis│                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Management                         │
├─────────────────────────────────────────────────────────────┤
│  • Model Configuration  • Performance Monitoring           │
│  • A/B Testing         • Logs & Analytics                  │
│  • Real-time Stats     • Research Tools                    │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
openwebui-setup/
├── README.md                           # This file
├── ARCHITECTURE.md                     # Detailed architecture docs
├── API_REFERENCE.md                    # API documentation
├── DEPLOYMENT.md                       # Deployment instructions
├── OPENWEBUI_INTEGRATION.md            # Open WebUI integration guide
├── docker-compose.yml                  # Docker orchestration
├── Dockerfile                          # Container definition
├── requirements.txt                    # Python dependencies
├── backend/                            # Backend and services
│   ├── api/                            # FastAPI management API
│   │   ├── main.py                     # API entrypoint (port 8095)
│   │   ├── pipelines.py                # Proxies to services
│   │   ├── models.py                   # Model management endpoints
│   │   ├── config.py                   # Config management endpoints
│   │   └── monitoring.py               # Monitoring and websockets
│   ├── models/                         # Data models
│   │   ├── llm_state.py
│   │   ├── model_config.py
│   │   ├── query_models.py
│   │   └── search_stats.py
│   ├── services/                       # Business logic services
│   │   ├── proper_treequest_ab_mcts_service.py  # AB-MCTS (port 8094)
│   │   ├── proper_multi_model_service.py        # Multi-Model (port 8090)
│   │   ├── experiment_logger.py                 # SQLite + JSONL runs
│   │   └── config_manager.py                    # Config management
│   ├── model_integration.py           # OpenAI-compatible model adapter (8098)
│   └── openwebui_integration.py       # Tool endpoints for Open WebUI (8097)
├── interfaces/                         # Static interfaces
│   ├── dashboard.html                  # Management dashboard (served on 8081)
│   ├── conversational_ab_mcts_interface.html
│   ├── real_ab_mcts_interface.html
│   └── tool_test.html
├── pipelines/                          # (Optional) pipeline artifacts
│   └── ab_mcts_pipeline.py
└── docs/                               # Additional docs
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Ollama with models: `deepseek-r1:1.5b`, `gemma3:1b`, `llama3.2:1b`

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd openwebui-setup

# Start services
docker-compose up -d

# Access interfaces
# Open WebUI: http://localhost:3000
# Management Dashboard: http://localhost:8081/dashboard.html
# Backend API docs: http://localhost:8095/api/docs
```

## 🔧 Services

| Service | Port | Description |
|---------|------|-------------|
| Open WebUI | 3000 | Main chat interface |
| AB-MCTS Service | 8094 | TreeQuest AB-MCTS implementation |
| Multi-Model Service | 8090 | Simple multi-model collaboration |
| Backend API | 8095 | Management dashboard API |
| MCP Server | 8096 | Tools bridge for Open WebUI |
| Open WebUI Integration | 8097 | Tool endpoints (OpenAPI) |
| Model Integration | 8098 | OpenAI-compatible model adapter |
| HTTP Server | 8081 | Static dashboard (`/dashboard.html`) |

## 📊 Current Status

### ✅ Completed
- [x] AB-MCTS implementation using TreeQuest
- [x] Multi-model collaboration service
- [x] Open WebUI integration as selectable models
- [x] Backend management dashboard
- [x] Anti-hallucination system
- [x] Docker containerization
- [x] Dynamic model selection and configuration
- [x] Real-time monitoring and analytics

### 🚧 Current Issues
- [ ] **Timeouts**: AB‑MCTS can exceed 300–600s on complex prompts; streaming keep‑alives mitigate UI timeouts but latency remains high
- [ ] **Verbosity**: AB‑MCTS responses can be overly long; needs length/structure controls
- [ ] **Quality drift**: Occasional hallucinations; add stricter validation/fact‑checking
- [ ] **Inconsistent model API**: `backend/api/models.py` mixes `ModelManager` with undefined `models_db/default_models`
- [ ] **Monitoring placeholders**: Some monitoring endpoints return mocked/aggregated data
- [ ] **Security hardening**: Auth/rate‑limits noted in docs but not fully enforced in code

### 📋 Next Priorities
- [ ] Fix timeout issues with streaming responses
- [ ] Optimize AB-MCTS performance and verbosity
- [ ] Improve response quality and accuracy
- [ ] Add loading indicators and better UX
- [ ] Implement response caching

## 🧾 Runs & Logging

- Where: `logs/` (shared volume). Structure:
  - `logs/runs.db` (SQLite index)
  - `logs/runs/YYYYMMDD/run_<id>.jsonl` (JSONL event stream per run)
- View in UI: `http://localhost:8081/dashboard.html` → “Runs” card
- API:
  - `GET http://localhost:8095/api/runs?limit=50`
  - `GET http://localhost:8095/api/runs/{run_id}`
  - `GET http://localhost:8095/api/runs/{run_id}/events?head=200`

## 🔬 Science Tools (optional)

Two helper tools are exposed for sanity checks in chemistry and materials:

- Chemistry: `chem_lipinski_pains` (RDKit-based)
  - Endpoint: `POST http://localhost:8097/tools/chem/lipinski_pains`
  - Body: `{ "smiles": "CCO" }`
  - RDKit is optional; if not installed in the image, the endpoint returns a clear error.

- Materials Project: `materials_project_lookup`
  - Endpoint: `POST http://localhost:8097/tools/materials/lookup`
  - Body: `{ "formula": "LiFePO4" }` or `{ "mp_id": "mp-149" }`
  - Requires `MATERIALS_PROJECT_API_KEY`.

Use these tools via:
- Open WebUI Tools (connect MCP server at `http://localhost:8096`) → tools appear in the Tools panel, or
- Direct HTTP requests, or
- From pipelines/services as sub-calls.

## ⚙️ Configuration Notes

- Set `MATERIALS_PROJECT_API_KEY` to enable Materials Project lookups.
- Logging directory can be controlled with `LOGS_DIR` (defaults to `/app/logs` in containers).
- Open WebUI model integration: either add `http://localhost:8098` as a Direct Connection (OpenAI‑compatible) or set `OPENAI_API_BASE_URLS` to include `http://model-integration:8098` in Docker.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Sakana AI](https://sakana.ai/) for the AB-MCTS research and TreeQuest library
- [Sakana AI AB-MCTS-ARC2](https://github.com/SakanaAI/ab-mcts-arc2) for the official implementation reference
- [Open WebUI](https://github.com/open-webui/open-webui) for the chat interface
- [Ollama](https://ollama.ai/) for local model serving
