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
├── docker-compose.yml                  # Docker orchestration
├── Dockerfile                          # Container definition
├── requirements.txt                    # Python dependencies
├── pipelines/                          # Open WebUI pipelines
│   ├── ab_mcts_pipeline.py            # AB-MCTS pipeline
│   └── multi_model_pipeline.py        # Multi-model pipeline
├── backend/                           # Backend management
│   ├── api/                          # FastAPI endpoints
│   ├── models/                       # Data models
│   ├── services/                     # Business logic
│   └── dashboard/                    # Web dashboard
├── services/                         # Standalone services
│   ├── ab_mcts_service.py           # AB-MCTS service
│   ├── multi_model_service.py       # Multi-model service
│   └── treequest_ab_mcts_service.py # TreeQuest implementation
└── interfaces/                       # User interfaces
    ├── conversational_ab_mcts_interface.html
    └── multi_model_interface.html
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
# AB-MCTS Interface: http://localhost:8080/conversational_ab_mcts_interface.html
# Backend Dashboard: http://localhost:8095
```

## 🔧 Services

| Service | Port | Description |
|---------|------|-------------|
| Open WebUI | 3000 | Main chat interface |
| AB-MCTS Service | 8094 | TreeQuest AB-MCTS implementation |
| Multi-Model Service | 8090 | Simple multi-model collaboration |
| Backend API | 8095 | Management dashboard API |
| HTTP Server | 8080 | Static file serving |

## 📊 Current Status

### ✅ Completed
- [x] Basic AB-MCTS implementation using TreeQuest
- [x] Multi-model collaboration service
- [x] Anti-hallucination system
- [x] Docker containerization
- [x] Basic web interfaces
- [x] Model selection and configuration

### 🚧 In Progress
- [ ] Open WebUI pipeline integration
- [ ] Backend management dashboard
- [ ] Real-time monitoring and analytics
- [ ] Performance optimization

### 📋 TODO
- [ ] Web search integration
- [ ] Advanced model management
- [ ] A/B testing framework
- [ ] Research tools and analytics
- [ ] Documentation completion

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
