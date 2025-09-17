# AB-MCTS & Multi-Model Pipeline Project - Current Status

## 🎯 **Project Goals & Scope**

### **Primary Objectives**
- Integrate Sakana AI's AB-MCTS (Adaptive Branching Monte Carlo Tree Search) with Open WebUI
- Implement Multi-Model collaboration system
- Create backend management dashboard for model configuration and monitoring
- Provide both tools as selectable models in Open WebUI interface

### **Current Architecture**
```
┌─────────────────────────────────────────────────────────────┐
│                    Open WebUI Interface                     │
│  Model Selection: ab-mcts | multi-model | other models     │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Model Integration Service                    │
│              (Port 8098 - localhost:8098)                  │
│  • /v1/models - Model discovery                            │
│  • /chat/completions - Chat completions                    │
│  • /models - Alternative model discovery                   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Services                        │
├─────────────────────────────────────────────────────────────┤
│  AB-MCTS Service (Port 8094)                               │
│  • TreeQuest implementation                                │
│  • Dynamic model selection                                 │
│  • Anti-hallucination prompts                             │
├─────────────────────────────────────────────────────────────┤
│  Multi-Model Service (Port 8090)                           │
│  • Collaborative AI responses                              │
│  • Model-specific prompting                                │
│  • Quality-based synthesis                                 │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Management Layer                         │
├─────────────────────────────────────────────────────────────┤
│  Backend API (Port 8095)                                   │
│  • Pipeline status monitoring                              │
│  • Model management                                        │
│  • Configuration endpoints                                 │
├─────────────────────────────────────────────────────────────┤
│  Management Dashboard (Port 8081)                          │
│  • Web-based configuration                                 │
│  • Real-time monitoring                                    │
│  • Model testing interface                                 │
└─────────────────────────────────────────────────────────────┘
```

## ✅ **Completed Features**

### **Core Functionality**
- [x] AB-MCTS service with TreeQuest implementation
- [x] Multi-Model collaboration service
- [x] Model discovery and dynamic selection
- [x] Open WebUI integration as selectable models
- [x] Backend API for management and monitoring
- [x] Management dashboard (web interface)
- [x] Docker containerization and orchestration
- [x] Anti-hallucination system
- [x] Model-specific prompting and configuration

### **Integration Status**
- [x] Models appear in Open WebUI model dropdown
- [x] Direct model selection (not tools)
- [x] Chat completions endpoint working
- [x] Model discovery working
- [x] Backend monitoring working

## 🚧 **Current Issues**

### **Critical Issues**
1. **Timeout Problems**
   - AB-MCTS service taking >120 seconds to respond
   - Current timeout: 300 seconds (still insufficient for complex queries)
   - Error: `httpx.ReadTimeout` in model integration service
   - **Impact**: Users get timeout errors in Open WebUI

2. **Performance Issues**
   - AB-MCTS responses are very verbose and complex
   - Response quality varies significantly
   - Some responses contain hallucinated information

### **Minor Issues**
1. **Dashboard Status**
   - Occasionally shows "unhealthy" due to timeout errors
   - Status updates may be delayed

2. **Model Configuration**
   - Default models: `smollm:135m`, `qwen3:0.6b`, `granite3.1-moe:1b`
   - May need optimization for better performance

## 🔧 **Technical Details**

### **Service Endpoints**
- **Model Integration**: `http://localhost:8098`
  - `/v1/models` - Model discovery
  - `/chat/completions` - Chat completions
  - `/models` - Alternative model discovery

- **AB-MCTS Service**: `http://localhost:8094`
  - `/query` - Process queries with tree search
  - `/models` - Get available models
  - `/models/update` - Update model selection

- **Multi-Model Service**: `http://localhost:8090`
  - `/query` - Process queries with collaboration
  - `/models` - Get available models
  - `/models/update` - Update model selection

- **Backend API**: `http://localhost:8095`
  - `/api/pipelines/status` - Pipeline health status
  - `/api/pipelines/ab-mcts/*` - AB-MCTS management
  - `/api/pipelines/multi-model/*` - Multi-Model management
  - `/api/runs/*` - Run listing and events
  - `/api/monitoring/*` - Metrics/logs/health, WebSocket at `/api/monitoring/ws`

### **Docker Services**
- `open-webui` - Main Open WebUI interface
- `ab-mcts-service` - AB-MCTS pipeline service
- `multi-model-service` - Multi-Model collaboration service
- `model-integration` - Open WebUI model integration
- `backend-api` - Management and monitoring API
- `http-server` - Static file serving for dashboards
 - `prometheus` - Metrics scraping
 - `grafana` - Dashboards (Prometheus datasource provisioned)

### **Network Configuration**
- Network: `openwebui-setup_openwebui-net`
- All services communicate via container names
- Open WebUI connects to model integration via `localhost:8098`

## 📁 **Project Structure**
```
openwebui-setup/
├── README.md                           # Main project documentation
├── PROJECT_STATUS.md                   # This file - current status
├── ARCHITECTURE.md                     # Detailed architecture docs
├── API_REFERENCE.md                    # API documentation
├── DEPLOYMENT.md                       # Deployment instructions
├── OPENWEBUI_INTEGRATION.md            # Integration guide
├── docker-compose.yml                  # Docker orchestration
├── Dockerfile                          # Container definition
├── requirements.txt                    # Python dependencies
├── backend/                           # Backend services
│   ├── api/                          # FastAPI endpoints
│   ├── models/                       # Data models
│   ├── services/                     # Business logic
│   ├── model_integration.py          # Open WebUI integration
│   └── openwebui_integration.py      # Tool integration (legacy)
├── interfaces/                       # Web interfaces
│   ├── dashboard.html                # Management dashboard
│   ├── tool_test.html               # Direct tool testing
│   └── index.html                    # Landing page
└── pipelines/                        # Open WebUI pipelines (legacy)
    └── ab_mcts_pipeline.py
```

## 🎯 **Next Steps**

### **Immediate Priorities**
1. **Long‑running AB‑MCTS**
   - Keep SSE streaming + periodic pings; ensure Open WebUI timeouts are ≥600s
   - Add adaptive iteration/max_depth based on prompt complexity
   - Expose per‑request overrides via model‑integration config
   - Consider partial result streaming/early finish on confidence

2. **Performance & Verbosity**
   - Enforce max length/sections; post‑summarize
   - Implement response caching
   - Add query classification for iteration selection
   - Optimize per‑model temps/prompts

3. **Stability & Consistency**
   - Refactor `backend/api/models.py` to remove legacy `models_db/default_models`
   - Replace mocked monitoring returns with real metrics or clearly mark as demo
   - Add auth/rate‑limits for public deployments

### **Future Enhancements**
1. **Advanced Features**
   - Web search integration
   - Real-time monitoring dashboard
   - A/B testing framework
   - Research tools and analytics

2. **Scalability**
   - Horizontal scaling
   - Load balancing
   - Resource optimization

## 🔑 **Key Configuration**

### **Open WebUI Setup**
1. Go to Settings → Connections
2. Enable "Direct Connections"
3. Add connection:
   - Name: `AB-MCTS & Multi-Model Models`
   - URL: `http://localhost:8098`
   - Auth: Bearer (no API key)

### **Model Selection**
- Models appear in Open WebUI model dropdown
- Select `ab-mcts` for complex problem solving
- Select `multi-model` for collaborative analysis

### **Management Access**
- Dashboard: `http://localhost:8081/dashboard.html`
- Tool Test: `http://localhost:8081/tool_test.html`
- Backend API: `http://localhost:8095/api/docs`
  - Metrics: `http://localhost:8095/metrics`
  - Research endpoints: `/api/monitoring/performance`, `/api/monitoring/passk`

## 📊 **Performance Metrics**
- AB‑MCTS: multi‑minute on complex queries
- Multi‑Model: 1–3 minutes
- Model Discovery: <1 second
- Health Checks: <1 second

## 🐛 **Known Issues**
1. Long AB‑MCTS latencies on complex prompts
2. Verbose responses (AB‑MCTS)
3. Occasional hallucinations
4. Monitoring endpoints partially mocked
5. Inconsistent models API (`backend/api/models.py` legacy refs)
6. Security hardening not fully implemented

## 📝 **Development Notes**
- All services use Docker containers
- Communication via HTTP APIs
- Model integration uses OpenAI-compatible endpoints
- Backend uses FastAPI with async/await
- Frontend uses vanilla JavaScript with modern CSS
