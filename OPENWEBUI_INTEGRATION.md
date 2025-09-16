# 🔗 Open WebUI Integration Guide

This guide explains how to integrate the AB-MCTS and Multi-Model services with Open WebUI.

## 🎯 Integration Methods

We provide multiple integration methods for Open WebUI:

### 1. **MCP Server Integration** (Recommended)
- **MCP Server**: `http://localhost:8096`
- **Protocol**: Model Context Protocol (MCP)
- **Tools Available**: 4 tools for AB-MCTS and Multi-Model management

### 2. **Direct OpenAPI Integration**
- **AB-MCTS API**: `http://localhost:8094` (OpenAPI spec available)
- **Multi-Model API**: `http://localhost:8090` (OpenAPI spec available)
- **Protocol**: OpenAPI 3.0

## 🚀 Quick Setup

### Method 1: MCP Server Integration

1. **Start the MCP Server** (already running):
   ```bash
   docker compose up -d mcp-server
   ```

2. **Configure Open WebUI**:
   - Open Open WebUI at `http://localhost:3000`
   - Go to Settings → Tools
   - Add new connection:
     - **Name**: AB-MCTS & Multi-Model Pipeline
     - **URL**: `http://localhost:8096`
     - **Type**: MCP Server

3. **Available Tools**:
   - `ab_mcts_query` - Run AB-MCTS queries
   - `multi_model_query` - Run Multi-Model queries
   - `ab_mcts_models` - Manage AB-MCTS models
   - `multi_model_models` - Manage Multi-Model models

### Method 2: Direct OpenAPI Integration

1. **Add AB-MCTS Service**:
   - Go to Settings → Tools
   - Add new connection:
     - **Name**: AB-MCTS Service
     - **URL**: `http://localhost:8094`
     - **Type**: OpenAPI

2. **Add Multi-Model Service**:
   - Go to Settings → Tools
   - Add new connection:
     - **Name**: Multi-Model Service
     - **URL**: `http://localhost:8090`
     - **Type**: OpenAPI

## 🛠️ Usage in Open WebUI

### Using AB-MCTS
1. Start a new chat in Open WebUI
2. The AB-MCTS tools will be available in the tools panel
3. Use `ab_mcts_query` to run complex reasoning tasks
4. Configure models with `ab_mcts_models`

### Using Multi-Model
1. Start a new chat in Open WebUI
2. The Multi-Model tools will be available in the tools panel
3. Use `multi_model_query` for collaborative AI responses
4. Configure models with `multi_model_models`

## 📊 Available Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| smollm:135m | 135M | ⚡⚡⚡⚡⚡ | 🎯🎯 | Fast testing |
| qwen3:0.6b | 0.6B | ⚡⚡⚡⚡ | 🎯🎯🎯 | Multilingual |
| granite3.1-moe:1b | 1B | ⚡⚡⚡⚡ | 🎯🎯🎯🎯 | Code tasks |
| llama3.2:1b | 1B | ⚡⚡⚡⚡ | 🎯🎯🎯 | Creative tasks |
| gemma3:1b | 1B | ⚡⚡⚡⚡ | 🎯🎯🎯 | Efficient responses |
| deepseek-r1:1.5b | 1.5B | ⚡⚡⚡⚡ | 🎯🎯🎯🎯 | Reasoning tasks |

## 🔧 Management Dashboard

For backend management, use the dedicated dashboard:
- **URL**: `http://localhost:8081/dashboard.html`
- **Purpose**: Model configuration, monitoring, and system management
- **Note**: This is for administrators only, not end users

## 🧪 Testing Integration

### Test MCP Server
```bash
curl -X POST http://localhost:8096/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": "1", "method": "tools/list"}'
```

### Test AB-MCTS Service
```bash
curl -X POST http://localhost:8094/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "iterations": 10, "max_depth": 3}'
```

### Test Multi-Model Service
```bash
curl -X POST http://localhost:8090/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

## 🔍 Troubleshooting

### Common Issues

1. **Tools not appearing in Open WebUI**:
   - Check if MCP server is running: `docker logs mcp-server`
   - Verify Open WebUI can reach the server
   - Check firewall settings

2. **Connection errors**:
   - Ensure all services are running: `docker compose ps`
   - Check service health: `curl http://localhost:8096/health`

3. **Model selection issues**:
   - Use the management dashboard to configure models
   - Check Ollama is running: `curl http://localhost:11434/api/tags`

### Debug Commands

```bash
# Check all services
docker compose ps

# Check service logs
docker logs mcp-server
docker logs ab-mcts-service
docker logs multi-model-service

# Test service connectivity
curl http://localhost:8096/health
curl http://localhost:8094/health
curl http://localhost:8090/health
```

## 📚 API Documentation

- **MCP Server**: `http://localhost:8096/docs`
- **AB-MCTS Service**: `http://localhost:8094/docs`
- **Multi-Model Service**: `http://localhost:8090/docs`
- **Backend API**: `http://localhost:8095/docs`

## 🎉 Success Indicators

When properly integrated, you should see:
- ✅ AB-MCTS and Multi-Model tools in Open WebUI
- ✅ Ability to run queries through Open WebUI interface
- ✅ Model management capabilities
- ✅ Real-time responses with search statistics

## 🔄 Next Steps

1. **Configure Models**: Use the management dashboard to select optimal models
2. **Test Queries**: Try different types of queries in Open WebUI
3. **Monitor Performance**: Use the backend dashboard for monitoring
4. **Customize**: Adjust parameters based on your use case

---

**Note**: This integration provides a seamless experience where users can access advanced AB-MCTS and Multi-Model capabilities directly through the familiar Open WebUI interface, while administrators can manage the system through the dedicated backend dashboard.
