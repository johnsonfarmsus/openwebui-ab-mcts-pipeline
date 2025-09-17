# API Reference

## 🔌 Service Endpoints

### AB-MCTS Service (Port 8094)

#### POST /query
Process a query using AB-MCTS algorithm.

**Request Body:**
```json
{
  "query": "How many chocolate chips must a cookie contain?",
  "iterations": 20,
  "max_depth": 5,
  "models": ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"],
  "conversation_id": "optional-uuid"
}
```

**Response:**
```json
{
  "result": "Based on general knowledge...",
  "success": true,
  "search_stats": {
    "total_iterations": 20,
    "nodes_created": 45,
    "best_reward": 0.85,
    "average_reward": 0.72,
    "exploration_ratio": 0.6,
    "width_searches": 12,
    "depth_searches": 8,
    "model_usage": {
      "deepseek-r1:1.5b": 8,
      "gemma3:1b": 7,
      "llama3.2:1b": 5
    },
    "model_used": "deepseek-r1:1.5b"
  },
  "conversation_id": "uuid",
  "turn_id": "uuid"
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### Multi-Model Service (Port 8090)

#### POST /query
Process a query using multi-model collaboration.

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "models": ["deepseek-r1:1.5b", "gemma3:1b", "llama3.2:1b"]
}
```

**Response:**
```json
{
  "result": "Machine learning is a subset of artificial intelligence...",
  "success": true,
  "search_stats": {
    "models_used": 3,
    "response_time": 2.5,
    "confidence": 0.9
  }
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### Backend Management API (Port 8095)

#### Models Management

##### GET /api/models
List all available models.

**Response:**
```json
[
  {
    "id": "deepseek-r1-1.5b",
    "name": "DeepSeek-R1 1.5B",
    "endpoint": "http://host.docker.internal:11434",
    "enabled": true,
    "performance_score": 0.85,
    "last_used": "2024-01-15T10:30:00Z"
  }
]
```

##### POST /api/models
Add a new model.

**Request Body:**
```json
{
  "name": "New Model",
  "endpoint": "http://localhost:11434",
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

##### PUT /api/models/{id}
Update model configuration.

**Request Body:**
```json
{
  "enabled": false,
  "parameters": {
    "temperature": 0.8
  }
}
```

##### DELETE /api/models/{id}
Remove a model.

#### Performance Monitoring

##### GET /api/monitoring/performance
Get performance metrics.

**Response:**
```json
{
  "ab_mcts": {
    "avg_response_time": 3.2,
    "success_rate": 0.95,
    "avg_score": 0.78
  },
  "multi_model": {
    "avg_response_time": 1.8,
    "success_rate": 0.98,
    "avg_confidence": 0.82
  },
  "models": {
    "deepseek-r1:1.5b": {
      "usage_count": 150,
      "avg_score": 0.85,
      "last_used": "2024-01-15T10:30:00Z"
    }
  }
}
```

##### GET /api/monitoring/logs
Get system logs.

**Query Parameters:**
- `level`: Log level (info, warning, error)
- `service`: Service name (ab_mcts, multi_model, backend)
- `limit`: Number of logs to return (default: 100)

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "level": "info",
      "service": "ab_mcts",
      "message": "Search completed with 20 iterations"
    }
  ],
  "total": 1000,
  "has_more": true
}
```

#### A/B Testing

##### POST /api/models/ab-test
Start an A/B test.

**Request Body:**
```json
{
  "name": "Model Comparison Test",
  "variants": [
    {
      "name": "AB-MCTS",
      "pipeline": "ab_mcts",
      "traffic_percentage": 50
    },
    {
      "name": "Multi-Model",
      "pipeline": "multi_model",
      "traffic_percentage": 50
    }
  ],
  "duration_hours": 24
}
```

##### GET /api/models/ab-test/{id}/results
Get A/B test results.

**Response:**
```json
{
  "test_id": "uuid",
  "status": "running",
  "results": {
    "variants": [
      {
        "name": "AB-MCTS",
        "requests": 150,
        "avg_score": 0.85,
        "success_rate": 0.95
      },
      {
        "name": "Multi-Model",
        "requests": 145,
        "avg_score": 0.78,
        "success_rate": 0.98
      }
    ]
  }
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://host.docker.internal:11434` |
| `WEBUI_SECRET_KEY` | Open WebUI secret key | `your-secret-key-here` |
| `BACKEND_API_URL` | Backend API URL | `http://localhost:8095` |
| `LOG_LEVEL` | Logging level | `info` |
| `MAX_ITERATIONS` | Maximum AB-MCTS iterations | `50` |
| `MAX_DEPTH` | Maximum search depth | `10` |

### Model Configuration

```json
{
  "models": [
    {
      "name": "deepseek-r1:1.5b",
      "endpoint": "http://host.docker.internal:11434",
      "parameters": {
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.9
      },
      "enabled": true
    }
  ]
}
```

## 📊 Error Codes

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_QUERY",
    "message": "Query cannot be empty",
    "details": {
      "field": "query",
      "value": ""
    }
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_QUERY` | Query is empty or invalid |
| `MODEL_NOT_FOUND` | Specified model not available |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `SEARCH_FAILED` | AB-MCTS search failed |
| `MODEL_ERROR` | Model API error |

## 🔍 Examples

### Basic AB-MCTS Query

```bash
curl -X POST http://localhost:8094/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the meaning of life?",
    "iterations": 10,
    "max_depth": 3
  }'
```

### Multi-Model Query

```bash
curl -X POST http://localhost:8090/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain quantum computing",
    "models": ["deepseek-r1:1.5b", "gemma3:1b"]
  }'
```

### Add New Model

```bash
curl -X POST http://localhost:8095/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Custom Model",
    "endpoint": "http://localhost:11434",
    "parameters": {
      "temperature": 0.8,
      "max_tokens": 1500
    }
  }'
```

### Get Performance Metrics

```bash
curl http://localhost:8095/api/monitoring/performance
```

## 🔄 WebSocket Events

### Real-time Monitoring

Connect to `ws://localhost:8095/api/monitoring/ws` for real-time updates.

#### Events

##### search_started
```json
{
  "event": "search_started",
  "data": {
    "query": "What is AI?",
    "pipeline": "ab_mcts",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

##### search_progress
```json
{
  "event": "search_progress",
  "data": {
    "iteration": 5,
    "best_score": 0.75,
    "nodes_created": 12,
    "current_model": "deepseek-r1:1.5b"
  }
}
```

##### search_completed
```json
{
  "event": "search_completed",
  "data": {
    "final_score": 0.85,
    "total_iterations": 20,
    "total_nodes": 45,
    "response_time": 3.2
  }
}
```

## 📝 Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/query` | 100 requests | 1 minute |
| `/api/models` | 50 requests | 1 minute |
| `/api/monitoring/performance` | 200 requests | 1 minute |
| `/api/monitoring/logs` | 30 requests | 1 minute |
