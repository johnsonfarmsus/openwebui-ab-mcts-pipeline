# Architecture Documentation

## 🏗️ System Architecture

### High-Level Overview

The system consists of three main layers:

1. **Presentation Layer**: Open WebUI chat interface with model selection
2. **Processing Layer**: AB-MCTS and Multi-Model pipelines
3. **Management Layer**: Backend API and monitoring dashboard

### Component Details

#### 1. Open WebUI Integration

**Purpose**: Provide native chat interface with seamless model selection

**Components**:
- **Pipeline System**: Custom pipelines for AB-MCTS and Multi-Model
- **Model Selection**: User can choose between different reasoning approaches
- **Web Search**: Integrated search capabilities for factual information

**Implementation**:
```python
# Pipeline structure
class ABMCTSPipeline:
    def __init__(self):
        self.treequest = TreeQuestABMCTS()
        self.anti_hallucination = AntiHallucinationSystem()
    
    def process(self, query, context):
        return self.treequest.search(query, context)
```

#### 2. AB-MCTS Pipeline

**Purpose**: Implement Sakana AI's AB-MCTS algorithm for advanced reasoning

**Key Features**:
- **Two-Dimensional Search**: Width (new solutions) and Depth (refinements)
- **Thompson Sampling**: Adaptive branching decisions
- **Multi-Model Collaboration**: Dynamic model selection
- **Anti-Hallucination**: Prevents fabricated information
- **Quality Scoring**: Evaluates solution quality

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

**Implementation**:
```python
class TreeQuestABMCTS:
    def __init__(self):
        self.algo = tq.ABMCTSA()
        self.models = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
    
    def search(self, query, iterations=20, max_depth=5):
        # TreeQuest implementation
        pass
```

#### 3. Multi-Model Pipeline

**Purpose**: Simple multi-model collaboration for fast responses

**Key Features**:
- **Direct Model Calls**: Parallel model execution
- **Response Synthesis**: Combine multiple model outputs
- **Fast Processing**: Minimal overhead
- **Model Voting**: Consensus-based decision making

**Implementation**:
```python
class MultiModelPipeline:
    def __init__(self):
        self.models = ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
    
    def process(self, query):
        responses = []
        for model in self.models:
            response = self.call_model(model, query)
            responses.append(response)
        return self.synthesize(responses)
```

#### 4. Backend Management

**Purpose**: Centralized management and monitoring

**Components**:
- **Model Manager**: Configure and manage models
- **Performance Monitor**: Real-time analytics
- **Configuration API**: Dynamic parameter adjustment
- **Research Tools**: Analysis and debugging

**API Endpoints**:
```
GET  /api/models              # List available models
POST /api/models              # Add new model
PUT  /api/models/{id}         # Update model config
GET  /api/performance         # Performance metrics
GET  /api/logs               # System logs
POST /api/ab-test            # A/B testing
```

## 🔄 Data Flow

### AB-MCTS Flow
```
User Query → Open WebUI → AB-MCTS Pipeline → TreeQuest → Model Calls → Quality Scoring → Best Solution → Response
```

### Multi-Model Flow
```
User Query → Open WebUI → Multi-Model Pipeline → Parallel Model Calls → Synthesis → Response
```

### Management Flow
```
Admin Dashboard → Backend API → Pipeline Configuration → Model Updates → Real-time Monitoring
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
    command: ["python", "treequest_ab_mcts_service.py"]
  
  multi-model-service:
    build: .
    ports: ["8090:8090"]
    command: ["python", "ab_mcts_service.py"]
  
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
- **Search Tree Visualization**: Live view of AB-MCTS tree
- **Performance Charts**: Response times and success rates
- **Model Comparison**: A/B testing results
- **Error Monitoring**: Real-time error tracking

## 🔒 Security Considerations

### Anti-Hallucination
- **Prompt Engineering**: Explicit instructions against fabrication
- **Scoring Penalties**: Deduct points for suspicious content
- **Content Filtering**: Detect and flag potential hallucinations

### API Security
- **Authentication**: JWT tokens for API access
- **Rate Limiting**: Prevent abuse of expensive operations
- **Input Validation**: Sanitize user inputs

## 🚀 Performance Optimization

### Caching
- **Response Caching**: Cache similar queries
- **Model Caching**: Keep models in memory
- **Tree Caching**: Persist search trees for similar queries

### Scaling
- **Horizontal Scaling**: Multiple pipeline instances
- **Load Balancing**: Distribute requests across instances
- **Async Processing**: Non-blocking model calls

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
