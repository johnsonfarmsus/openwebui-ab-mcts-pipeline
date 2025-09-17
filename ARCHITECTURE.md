# Architecture Documentation

## 🏗️ System Architecture

### High-Level Overview

The system consists of three main layers:

1. **Presentation Layer**: Open WebUI chat interface with model selection
2. **Processing Layer**: AB‑MCTS and Multi‑Model services (FastAPI)
3. **Management Layer**: Backend API (management, monitoring, runs) and static dashboard

### Component Details

#### 1. Open WebUI Integration

**Purpose**: Provide native chat interface with seamless model selection

**Components**:
- **Model Integration** (`backend/model_integration.py`, port 8098): OpenAI‑compatible models `ab-mcts` and `multi-model` exposed via `/v1/models` and `/v1/chat/completions`, with streaming and keep‑alive for long‑running queries.
- **Tools Integration** (`backend/openwebui_integration.py`, port 8097): OpenAPI endpoints to call AB‑MCTS and Multi‑Model directly as tools; includes optional science tools (RDKit Lipinski/PAINS, Materials Project lookup).
- **MCP Server** (`backend/mcp_server.py`, port 8096): Exposes the above tools via Model Context Protocol for Open WebUI Tools panel.

**Notes**:
- Open WebUI can connect to `model-integration` as a Direct Connection (OpenAI API) or discover tools via MCP/HTTP.
- The legacy “pipeline system” is not used in this implementation; services run as dedicated FastAPI apps.

#### 2. AB‑MCTS Service

**Purpose**: Implement Sakana AI's AB‑MCTS algorithm (TreeQuest) for advanced reasoning

**Key Features**:
- **Two-Dimensional Search**: Width (new solutions) and Depth (refinements)
- **TreeQuest ABMCTSA**: Iterative width/depth expansion with model‑specific generation
- **Width/Depth Prompts**: Structured prompts for new solutions vs. refinements
- **Quality Scoring**: Length/structure/relevance/confidence composite
- **Iteration Log**: Optional per‑iteration snapshots for visualization

**Algorithm Flow**:
```
1. Initialize root node
2. For each iteration:
   a. Selection: Choose best path using UCB1
   b. Expansion: Generate new actions (width/depth)
   c. Simulation: Get model response and score
   d. Backpropagation: Update node statistics
3. Return best solution
```

Implementation: `backend/services/proper_treequest_ab_mcts_service.py` (port 8094)

#### 3. Multi‑Model Service

**Purpose**: Model‑aware parallel collaboration and synthesis for fast, comprehensive responses

**Key Features**:
- **Model Discovery**: Detect available Ollama models dynamically
- **Model‑Specific Prompting**: Tailored prompts per model strengths
- **Quality Scoring**: Relevance/structure/length assessment
- **Synthesis**: Uses best model to synthesize combined answer (fallback to weighted merge)

Implementation: `backend/services/proper_multi_model_service.py` (port 8090)

#### 4. Backend Management API

**Purpose**: Centralized management and monitoring

**Components**:
- **Pipelines Router**: `/api/pipelines` proxies to AB‑MCTS and Multi‑Model services
- **Models Router**: `/api/models/*` model CRUD/testing/performance/A/B tests
- **Config Router**: `/api/config/*` dynamic configs, backup/restore
- **Monitoring Router**: `/api/monitoring/*` metrics, logs, health, WebSocket `/api/monitoring/ws`
- **Runs API**: `/api/runs/*` list/get run logs (via `ExperimentLogger`)

**Key Endpoints** (selected):
```
GET  /api/pipelines/status
POST /api/pipelines/ab-mcts/query
POST /api/pipelines/multi-model/query

GET  /api/models/
GET  /api/models/{model_id}
POST /api/models/                     # create
PUT  /api/models/{model_id}           # update
POST /api/models/{model_id}/test
GET  /api/models/{model_id}/performance
POST /api/models/ab-test
GET  /api/models/ab-test/{test_id}/results

GET  /api/config/
PUT  /api/config/{key}
GET  /api/config/health

GET  /api/monitoring/performance
GET  /api/monitoring/logs
GET  /api/monitoring/metrics
GET  /api/monitoring/health
WS   /api/monitoring/ws

GET  /api/runs
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/events
```

## 🔄 Data Flow

### AB‑MCTS Flow
```
User → Open WebUI (model: ab-mcts) → Model Integration → AB‑MCTS Service (TreeQuest) → Ollama Models → Best Solution → Streamed back
```

### Multi‑Model Flow
```
User → Open WebUI (model: multi-model) → Model Integration → Multi‑Model Service → Parallel Ollama Calls → Synthesis → Streamed back
```

### Management Flow
```
Admin (dashboard/http) → Backend API → Pipelines/Models/Config → Runs & Monitoring (WebSocket)
```

## 🗄️ Data Models

### LLMState
```python
@dataclass
class LLMState:
    answer: str
    model_used: str
    action_type: str  # "width" or "depth"
    score: float
    depth: int
```

### SearchStats
```python
@dataclass
class SearchStats:
    total_iterations: int
    nodes_created: int
    best_reward: float
    average_reward: float
    exploration_ratio: float
    width_searches: int
    depth_searches: int
    model_usage: Dict[str, int]
```

### ModelConfig
```python
@dataclass
class ModelConfig:
    name: str
    endpoint: str
    parameters: Dict[str, Any]
    enabled: bool
    performance_score: float
```

## 🔧 Configuration

### Environment Variables
```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
WEBUI_SECRET_KEY=your-secret-key
BACKEND_API_URL=http://localhost:8095
```

### Docker Compose Services
```yaml
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports: ["3000:8080"]
  
  ab-mcts-service:
    build: .
    ports: ["8094:8094"]
    command: ["python", "backend/services/proper_treequest_ab_mcts_service.py"]
  
  multi-model-service:
    build: .
    ports: ["8090:8090"]
    command: ["python", "backend/services/proper_multi_model_service.py"]
  
  backend-api:
    build: .
    ports: ["8095:8095"]
    command: ["python", "backend/api/main.py"]
```

## 📊 Monitoring & Analytics

### Metrics Tracked
- **Search Performance**: Iterations, nodes created, scores
- **Model Usage**: Which models are selected most often
- **Response Quality**: User satisfaction and accuracy
- **System Health**: API response times, error rates

### Real-time Dashboard
- **Runs**: View past runs and event streams (via `ExperimentLogger`)
- **Performance**: Response times, success rates (monitoring endpoints)
- **Health**: Service statuses and periodic updates (WebSocket)

## 🔒 Security Considerations

### Anti-Hallucination
- **Prompt Engineering**: Explicit instructions against fabrication
- **Scoring Penalties**: Deduct points for suspicious content
- **Content Filtering**: Detect and flag potential hallucinations

### API Security
- Authentication/authorization and rate limiting are planned; current code does not fully enforce them. Input validation is handled at Pydantic model boundaries.

## 🚀 Performance Optimization

### Caching
- **Response Caching**: Cache similar queries
- **Model Caching**: Keep models in memory
- **Tree Caching**: Persist search trees for similar queries

### Scaling
- **Horizontal Scaling**: Multiple instances of services (compose/k8s)
- **Load Balancing**: Front by gateway/proxy
- **Async Processing**: Non-blocking httpx/requests

## ⚠️ Current Limitations

- AB‑MCTS queries may take several minutes; streaming keep‑alives are implemented but latency remains high for complex prompts.
- Responses can be overly verbose; length/structure controls are not enforced universally.
- Occasional hallucinations remain; fact‑checking/validation is basic.
- `backend/api/models.py` contains legacy references (`models_db/default_models`) mixed with `ModelManager` usage.
- Some monitoring endpoints return aggregated/mock data for demo purposes.
- Security hardening (auth/rate limits) is not fully implemented.

## 🔄 Deployment Strategy

### Development
```bash
docker-compose up -d
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Monitoring
```bash
docker-compose logs -f
```

## 📈 Future Enhancements

### Planned Features
- **Web Search Integration**: Real-time fact checking
- **Advanced Analytics**: Machine learning insights
- **Custom Algorithms**: User-defined search strategies
- **API Marketplace**: Third-party integrations

### Research Areas
- **Algorithm Optimization**: Improve search efficiency
- **Model Selection**: Better model choice strategies
- **Quality Metrics**: More accurate scoring functions
- **User Experience**: Enhanced interfaces and workflows
