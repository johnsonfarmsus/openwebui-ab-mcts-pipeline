# Project Handoff Prompt for AB-MCTS & Multi-Model Pipeline

## 🎯 **Project Overview**

I'm working on integrating Sakana AI's AB-MCTS (Adaptive Branching Monte Carlo Tree Search) and a Multi-Model collaboration system with Open WebUI. The project creates advanced AI reasoning capabilities that appear as selectable models in Open WebUI's interface.

## 🏗️ **Current Architecture**

### **System Components**
- **Open WebUI**: Main chat interface with model selection dropdown
- **Model Integration Service** (Port 8098): OpenAI‑compatible models (ab‑mcts, multi‑model) with streaming/keep‑alive
- **AB‑MCTS Service** (Port 8094): TreeQuest‑based tree search for complex problem solving
- **Multi‑Model Service** (Port 8090): Collaborative AI using multiple models with synthesis
- **Open WebUI Tools Service** (Port 8097): Tool endpoints for AB‑MCTS/Multi‑Model + science tools
- **MCP Server** (Port 8096): MCP wrapper exposing tools to Open WebUI
- **Backend API** (Port 8095): Management/monitoring/runs endpoints
- **Management Dashboard** (Port 8081): Static web dashboard (served via http-server)

### **Integration Flow**
```
Open WebUI (Direct Connection) → Model Integration (8098) → AB‑MCTS / Multi‑Model → Ollama
Open WebUI (Tools)            → MCP (8096) / Tools (8097) → AB‑MCTS / Multi‑Model → Ollama
```

## ✅ **What's Working**

### **Completed Features**
- [x] AB-MCTS service with TreeQuest implementation
- [x] Multi-Model collaboration service  
- [x] Open WebUI integration as selectable models
- [x] Dynamic model discovery and selection
- [x] Backend management API
- [x] Web-based management dashboard
- [x] Docker containerization
- [x] Anti-hallucination system

### **Current Status**
- Models appear in Open WebUI model dropdown
- Users can select `ab-mcts` or `multi-model` as chat models
- Backend services are running and healthy
- Management dashboard is functional

## 🚧 **Critical Issues to Address**

### **1. Long‑running AB‑MCTS**
- **Issue**: Complex prompts can take minutes; model‑integration uses streaming keep‑alives but total latency remains high
- **Impact**: UX suffers; potential upstream gateway timeouts if misconfigured
- **Mitigation**: Streaming SSE with periodic pings; consider adaptive iteration limits, early‑exit heuristics

### **2. Verbosity & Response Control**
- **Issue**: AB‑MCTS responses can be overly long
- **Impact**: TL;DR responses and higher token costs
- **Mitigation**: Add stricter length/section caps in prompts; post‑processing summarization

### **3. Response Quality / Hallucinations**
- **Issue**: Occasional hallucinations in both AB‑MCTS and Multi‑Model
- **Mitigation**: Improve validation/fact‑checks; penalize speculation; integrate web search if feasible

### **4. API Consistency (Models router)**
- **Issue**: `backend/api/models.py` mixes `ModelManager` with undefined `models_db/default_models`
- **Impact**: Some endpoints may not function or reflect runtime model state
- **Suggestion**: Refactor to consistently use `ModelManager` and remove legacy variables

### **5. Monitoring Data**
- **Issue**: Some monitoring endpoints return mocked/aggregated data
- **Impact**: Dashboard metrics may not reflect real runtime
- **Suggestion**: Wire to real metrics or clearly flag as demo

### **6. Security Hardening**
- **Issue**: Auth/rate‑limits referenced but not enforced broadly
- **Impact**: Potential exposure in multi‑tenant or public deployments
- **Suggestion**: Add auth middleware, tokens, and rate‑limiters on critical endpoints

## 🔧 **Technical Details**

### **Key Files**
- `backend/model_integration.py` - Open WebUI integration service
- `backend/services/proper_treequest_ab_mcts_service.py` - AB-MCTS implementation
- `backend/services/proper_multi_model_service.py` - Multi-Model implementation
- `docker-compose.yml` - Service orchestration
- `interfaces/dashboard.html` - Management interface

### **Service URLs**
- Model Integration: `http://localhost:8098`
- AB-MCTS Service: `http://localhost:8094`
- Multi-Model Service: `http://localhost:8090`
- Backend API: `http://localhost:8095`
- Management Dashboard: `http://localhost:8081/dashboard.html`
 - Prometheus: `http://localhost:9090`
 - Grafana: `http://localhost:3001`

### **Open WebUI Configuration**
- Settings → Connections → Direct Connections → add `http://localhost:8098`
- Or Tools via MCP (8096) / HTTP tools (8097)
 - Observability: Prometheus scrapes `/metrics` on backend‑api (8095), ab‑mcts (8094), multi‑model (8090). Grafana datasource is provisioned by default.

## 🎯 **Immediate Next Steps**

### **Priority 1: Long‑running AB‑MCTS**
1. Keep SSE streaming + pings; ensure Open WebUI timeouts are ≥600s (compose already sets)
2. Add adaptive iteration/max_depth based on prompt complexity
3. Expose per‑request overrides via model‑integration config
4. Consider partial result streaming and early finish on confidence

### **Priority 2: Performance & Verbosity**
1. Enforce max length/sections; summarize in post‑processing
2. Add response caching keyed by normalized prompt
3. Add query classification (simple vs complex) to set iterations
4. Optimize per‑model temps/prompts

### **Priority 3: UX & Quality**
1. Loading indicators are present via streaming; improve progress texts
2. Add structured `iteration_log` rendering in dashboard
3. Harden error handling in model‑integration (already returns apologies)
4. Add fact‑checking hooks (optional web search)

## 📊 **Current Performance**
- AB‑MCTS: multi‑minute on complex prompts (optimize)
- Multi‑Model: typically 1–3 minutes
- Model Discovery: <1 second
- Health Checks: <1 second

## 🐛 **Known Issues**
1. Long AB‑MCTS latencies on complex prompts
2. Verbose responses (AB‑MCTS)
3. Occasional hallucinations
4. Monitoring endpoints partially mocked
5. Inconsistent models API (`backend/api/models.py` legacy refs)
6. Security hardening not fully implemented

## 🔍 **Debugging Commands**

### **Check Service Status**
```bash
# Check all services
docker ps

# Check AB-MCTS service
curl -s http://localhost:8094/health

# Check model integration
curl -s http://localhost:8098/health

# Test AB-MCTS query
curl -s -X POST http://localhost:8098/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ab-mcts", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

### **Check Logs**
```bash
# Model integration logs
docker logs model-integration --tail 20

# AB-MCTS service logs
docker logs ab-mcts-service --tail 20

# Backend API logs
docker logs backend-api --tail 20
```

## 📁 **Project Structure**
```
openwebui-setup/
├── README.md                           # Main documentation
├── PROJECT_STATUS.md                   # Current status
├── PROJECT_HANDOFF_PROMPT.md          # This file
├── docker-compose.yml                  # Docker orchestration
├── requirements.txt                    # Python dependencies
├── backend/                           # Backend services
│   ├── model_integration.py          # Open WebUI integration
│   ├── services/                     # Core services
│   └── api/                          # Management API
├── interfaces/                       # Web interfaces
│   ├── dashboard.html                # Management dashboard
│   └── tool_test.html               # Direct testing
└── pipelines/                        # Legacy Open WebUI pipelines
```

## 🎯 **Success Criteria**
1. AB-MCTS responses complete within 2-3 minutes
2. Responses are concise and accurate
3. No timeout errors in Open WebUI
4. Smooth user experience with loading indicators
5. Reliable model selection and switching

## 💡 **Suggested Approaches**

### **For Timeout Issues**
1. Implement streaming responses
2. Add query complexity detection
3. Use smaller, faster models for simple queries
4. Implement response caching

### **For Performance Issues**
1. Reduce AB-MCTS iterations for simple queries
2. Implement response length limits
3. Add query classification (simple vs complex)
4. Optimize model selection

### **For Quality Issues**
1. Improve anti-hallucination prompts
2. Add response validation
3. Implement quality scoring
4. Add fact-checking mechanisms

## 🔗 **Key Resources**
- Sakana AI AB-MCTS: https://github.com/SakanaAI/ab-mcts-arc2
- TreeQuest Library: https://github.com/SakanaAI/treequest
- TreeQuest PyPI: https://pypi.org/project/treequest/
- Open WebUI: https://github.com/open-webui/open-webui

## 📝 **Notes**
- All services are containerized with Docker
- Communication via HTTP APIs
- Model integration uses OpenAI-compatible endpoints
- Backend uses FastAPI with async/await
- Frontend uses vanilla JavaScript with modern CSS
- Current models: `smollm:135m`, `qwen3:0.6b`, `granite3.1-moe:1b`

This project is functional but needs optimization for production use. The core architecture is solid, but performance and user experience need significant improvement.
