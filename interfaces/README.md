# 🚀 AB-MCTS & Multi-Model Pipeline Dashboard

A modern web interface for managing and interacting with the AB-MCTS and Multi-Model pipeline system.

## 🌐 Access URLs

- **Main Dashboard**: http://localhost:8081/dashboard.html
- **Landing Page**: http://localhost:8081/
- **Open WebUI**: http://localhost:3000
- **API Documentation**: http://localhost:8095/docs

## ✨ Features

### 🧠 AB-MCTS Configuration
- **Model Selection**: Choose from 6+ available models
- **Parameter Tuning**: Adjust iterations and max depth
- **Real-time Testing**: Test model combinations before use
- **Performance Monitoring**: View width/depth search statistics

### 🤝 Multi-Model Configuration
- **Model Selection**: Select models for collaboration
- **Quality Scoring**: View model performance ratings
- **Collaboration Testing**: Test multi-model responses
- **Synthesis Monitoring**: Monitor response quality

### 💬 Query Interface
- **Pipeline Selection**: Choose between AB-MCTS or Multi-Model
- **Real-time Queries**: Run queries with live results
- **Performance Stats**: View search statistics and timing
- **Response Analysis**: Analyze model responses and quality

### 📊 System Monitoring
- **Service Status**: Real-time health monitoring
- **Model Discovery**: Automatic detection of available models
- **Performance Tracking**: Monitor response times and quality
- **Error Handling**: Comprehensive error reporting

## 🎯 Available Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| smollm:135m | 135M | ⚡⚡⚡⚡⚡ | 🎯🎯 | Fast testing |
| qwen3:0.6b | 0.6B | ⚡⚡⚡⚡ | 🎯🎯🎯 | Multilingual |
| granite3.1-moe:1b | 1B | ⚡⚡⚡⚡ | 🎯🎯🎯🎯 | Code tasks |
| llama3.2:1b | 1B | ⚡⚡⚡⚡ | 🎯🎯🎯 | Creative tasks |
| gemma3:1b | 1B | ⚡⚡⚡⚡ | 🎯🎯🎯 | Efficient responses |
| deepseek-r1:1.5b | 1.5B | ⚡⚡⚡⚡ | 🎯🎯🎯🎯 | Reasoning tasks |

## 🚀 Quick Start

1. **Open Dashboard**: Navigate to http://localhost:8081/dashboard.html
2. **Select Models**: Choose models for AB-MCTS and Multi-Model pipelines
3. **Configure Parameters**: Set iterations, depth, and other parameters
4. **Test Models**: Use the test buttons to verify model combinations
5. **Run Queries**: Use the query interface to test the system

## 🔧 Configuration

### AB-MCTS Parameters
- **Iterations**: Number of search iterations (1-100)
- **Max Depth**: Maximum tree depth (1-20)
- **Models**: Selected models for the search

### Multi-Model Parameters
- **Models**: Selected models for collaboration
- **Quality Threshold**: Minimum quality score for responses

## 📈 Performance Tips

1. **For Testing**: Use smaller models (SmolLM, Qwen) for faster responses
2. **For Production**: Use larger models (DeepSeek, Gemma) for better quality
3. **For Speed**: Lower iterations and depth for faster responses
4. **For Quality**: Higher iterations and depth for better results

## 🛠️ Troubleshooting

### Common Issues
- **Service Unavailable**: Check if Docker containers are running
- **Model Not Found**: Ensure Ollama is running and models are pulled
- **Timeout Errors**: Increase timeout values in the configuration
- **CORS Errors**: Ensure services are running on correct ports

### Debug Steps
1. Check system status on the dashboard
2. Verify model availability
3. Test individual services
4. Check Docker container logs

## 🔗 API Endpoints

### System Status
- `GET /api/pipelines/status` - System health check

### AB-MCTS
- `GET /models` - Get available models
- `POST /models/update` - Update selected models
- `POST /models/test` - Test model combination
- `POST /query` - Run AB-MCTS query

### Multi-Model
- `GET /models` - Get available models
- `POST /models/update` - Update selected models
- `POST /models/test` - Test model combination
- `POST /query` - Run Multi-Model query

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

See CONTRIBUTING.md for contribution guidelines.

## 🙏 Acknowledgments

- [Sakana AI](https://sakana.ai/) for AB-MCTS research and TreeQuest library
- [Open WebUI](https://github.com/open-webui/open-webui) for the chat interface
- [Ollama](https://ollama.ai/) for local model serving
