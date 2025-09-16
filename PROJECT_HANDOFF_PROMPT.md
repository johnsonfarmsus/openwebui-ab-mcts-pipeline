# Project Handoff Prompt for AB-MCTS & Multi-Model Pipeline

## 🎯 **Project Overview**

I'm working on integrating Sakana AI's AB-MCTS (Adaptive Branching Monte Carlo Tree Search) and a Multi-Model collaboration system with Open WebUI. The project creates advanced AI reasoning capabilities that appear as selectable models in Open WebUI's interface.

## 🏗️ **Current Architecture**

### **System Components**
- **Open WebUI**: Main chat interface with model selection dropdown
- **Model Integration Service** (Port 8098): Bridges Open WebUI with backend services
- **AB-MCTS Service** (Port 8094): TreeQuest-based tree search for complex problem solving
- **Multi-Model Service** (Port 8090): Collaborative AI using multiple models
- **Backend API** (Port 8095): Management and monitoring endpoints
- **Management Dashboard** (Port 8081): Web-based configuration interface

### **Integration Flow**
```
Open WebUI → Model Integration Service → Backend Services → Ollama Models
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

### **1. Timeout Problems (URGENT)**
- **Issue**: AB-MCTS service taking >300 seconds to respond
- **Error**: `httpx.ReadTimeout` in model integration service
- **Impact**: Users get timeout errors in Open WebUI
- **Current Timeout**: 300 seconds (still insufficient)
- **Location**: `backend/model_integration.py` line 136

### **2. Performance Issues**
- **Issue**: AB-MCTS responses are extremely verbose and complex
- **Impact**: Poor user experience, slow responses
- **Location**: AB-MCTS service response generation

### **3. Response Quality**
- **Issue**: Some responses contain hallucinated information
- **Impact**: Unreliable outputs
- **Location**: Model prompting and response synthesis

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

### **Open WebUI Configuration**
- Settings → Connections → Direct Connections
- URL: `http://localhost:8098`
- Auth: Bearer (no API key)

## 🎯 **Immediate Next Steps**

### **Priority 1: Fix Timeout Issues**
1. Increase timeout to 600+ seconds
2. Implement streaming responses
3. Add progress indicators in Open WebUI
4. Consider reducing AB-MCTS complexity

### **Priority 2: Optimize Performance**
1. Reduce response verbosity
2. Implement response caching
3. Add query complexity detection
4. Optimize model selection

### **Priority 3: Improve User Experience**
1. Add loading indicators
2. Implement response streaming
3. Better error handling
4. Response quality improvements

## 📊 **Current Performance**
- AB-MCTS: 5-10 minutes for complex queries (too slow)
- Multi-Model: 1-3 minutes (acceptable)
- Model Discovery: <1 second
- Health Checks: <1 second

## 🐛 **Known Issues**
1. Timeout errors with complex AB-MCTS queries
2. Extremely verbose AB-MCTS responses
3. Occasional hallucination in responses
4. Dashboard status updates may be delayed

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
- Open WebUI: https://github.com/open-webui/open-webui
- Project Repository: https://github.com/johnsonfarmsus/ab-mcts-arc2

## 📝 **Notes**
- All services are containerized with Docker
- Communication via HTTP APIs
- Model integration uses OpenAI-compatible endpoints
- Backend uses FastAPI with async/await
- Frontend uses vanilla JavaScript with modern CSS
- Current models: `smollm:135m`, `qwen3:0.6b`, `granite3.1-moe:1b`

This project is functional but needs optimization for production use. The core architecture is solid, but performance and user experience need significant improvement.
