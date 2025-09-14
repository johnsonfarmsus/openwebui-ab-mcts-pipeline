# Deployment Guide

## 🚀 Quick Start

### Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Ollama**: Latest version with required models
- **System Requirements**: 16GB RAM, 4 CPU cores (minimum)

### Required Models

Install the following models in Ollama:

```bash
# Install required models
ollama pull deepseek-r1:1.5b
ollama pull gemma3:1b
ollama pull llama3.2:1b

# Verify installation
ollama list
```

### 1. Clone Repository

```bash
git clone <repository-url>
cd openwebui-setup
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Environment Variables:**
```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://host.docker.internal:11434

# Open WebUI Configuration
WEBUI_SECRET_KEY=your-secret-key-here

# Backend API Configuration
BACKEND_API_URL=http://localhost:8095

# Logging Configuration
LOG_LEVEL=info

# AB-MCTS Configuration
MAX_ITERATIONS=50
MAX_DEPTH=10
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Verify Installation

```bash
# Check Open WebUI
curl http://localhost:3000/health

# Check AB-MCTS Service
curl http://localhost:8094/health

# Check Multi-Model Service
curl http://localhost:8090/health

# Check Backend API
curl http://localhost:8095/health
```

## 🐳 Docker Configuration

### Docker Compose Services

```yaml
version: '3.8'

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - WEBUI_SECRET_KEY=${WEBUI_SECRET_KEY}
    volumes:
      - open-webui:/app/backend/data
    networks:
      - openwebui-net
    restart: always

  ab-mcts-service:
    build: .
    container_name: ab-mcts-service
    ports:
      - "8094:8094"
    environment:
      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL}
      - LOG_LEVEL=${LOG_LEVEL}
    networks:
      - openwebui-net
    restart: always
    command: ["python", "treequest_ab_mcts_service.py"]

  multi-model-service:
    build: .
    container_name: multi-model-service
    ports:
      - "8090:8090"
    environment:
      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL}
      - LOG_LEVEL=${LOG_LEVEL}
    networks:
      - openwebui-net
    restart: always
    command: ["python", "ab_mcts_service.py"]

  backend-api:
    build: .
    container_name: backend-api
    ports:
      - "8095:8095"
    environment:
      - BACKEND_API_URL=${BACKEND_API_URL}
      - LOG_LEVEL=${LOG_LEVEL}
    networks:
      - openwebui-net
    restart: always
    command: ["python", "backend/api/main.py"]

networks:
  openwebui-net:
    driver: bridge

volumes:
  open-webui:
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8090

# Run the service
CMD ["python", "main.py"]
```

## 🔧 Production Deployment

### 1. Production Environment

```bash
# Create production environment file
cp .env.example .env.prod

# Edit production configuration
nano .env.prod
```

**Production Environment Variables:**
```bash
# Security
WEBUI_SECRET_KEY=your-production-secret-key
JWT_SECRET=your-jwt-secret

# Performance
MAX_ITERATIONS=100
MAX_DEPTH=15
LOG_LEVEL=warning

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

### 2. Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui-prod
    ports:
      - "80:8080"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - WEBUI_SECRET_KEY=${WEBUI_SECRET_KEY}
    volumes:
      - open-webui-prod:/app/backend/data
    networks:
      - openwebui-net
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  ab-mcts-service:
    build: .
    container_name: ab-mcts-service-prod
    ports:
      - "8094:8094"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - LOG_LEVEL=${LOG_LEVEL}
    networks:
      - openwebui-net
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  # ... other services
```

### 3. Start Production Services

```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale ab-mcts-service=3
```

## 📊 Monitoring & Logging

### 1. Log Management

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f ab-mcts-service

# View logs with timestamps
docker-compose logs -f --timestamps

# Follow logs in real-time
docker-compose logs -f --tail=100
```

### 2. Health Checks

```bash
# Check service health
curl http://localhost:3000/health
curl http://localhost:8094/health
curl http://localhost:8090/health
curl http://localhost:8095/health

# Check Docker container status
docker-compose ps
docker stats
```

### 3. Performance Monitoring

```bash
# Monitor resource usage
docker stats

# Check container logs for errors
docker-compose logs | grep ERROR

# Monitor API performance
curl http://localhost:8095/api/performance
```

## 🔄 Updates & Maintenance

### 1. Update Services

```bash
# Pull latest images
docker-compose pull

# Rebuild services
docker-compose build

# Restart services
docker-compose up -d
```

### 2. Backup & Restore

```bash
# Backup Open WebUI data
docker run --rm -v open-webui:/data -v $(pwd):/backup alpine tar czf /backup/open-webui-backup.tar.gz -C /data .

# Restore Open WebUI data
docker run --rm -v open-webui:/data -v $(pwd):/backup alpine tar xzf /backup/open-webui-backup.tar.gz -C /data
```

### 3. Cleanup

```bash
# Remove unused containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check logs
docker-compose logs service-name

# Check port conflicts
netstat -tulpn | grep :PORT

# Restart service
docker-compose restart service-name
```

#### 2. Model Not Found

```bash
# Check Ollama models
ollama list

# Pull missing model
ollama pull model-name

# Restart service
docker-compose restart ab-mcts-service
```

#### 3. Memory Issues

```bash
# Check memory usage
docker stats

# Increase memory limits
# Edit docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

#### 4. Network Issues

```bash
# Check network connectivity
docker network ls
docker network inspect openwebui-setup_openwebui-net

# Restart network
docker-compose down
docker-compose up -d
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=debug
docker-compose up -d

# Run service in debug mode
docker-compose run --rm ab-mcts-service python -m pdb treequest_ab_mcts_service.py
```

## 🔒 Security Considerations

### 1. Environment Security

```bash
# Use strong secrets
openssl rand -base64 32

# Restrict file permissions
chmod 600 .env

# Use Docker secrets
echo "your-secret" | docker secret create webui_secret -
```

### 2. Network Security

```yaml
# Restrict network access
networks:
  openwebui-net:
    driver: bridge
    internal: true
```

### 3. Container Security

```dockerfile
# Run as non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser
```

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale AB-MCTS service
docker-compose up -d --scale ab-mcts-service=3

# Use load balancer
# Add nginx or traefik configuration
```

### Vertical Scaling

```yaml
# Increase resource limits
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to production
        run: |
          docker-compose -f docker-compose.prod.yml up -d
```

## 📞 Support

### Getting Help

1. **Check Logs**: Always check service logs first
2. **Documentation**: Refer to API_REFERENCE.md
3. **Issues**: Create GitHub issue with logs
4. **Community**: Join Discord/Telegram for help

### Useful Commands

```bash
# Quick health check
curl -s http://localhost:3000/health && echo "Open WebUI OK"
curl -s http://localhost:8094/health && echo "AB-MCTS OK"
curl -s http://localhost:8090/health && echo "Multi-Model OK"
curl -s http://localhost:8095/health && echo "Backend OK"

# View all services
docker-compose ps

# Restart all services
docker-compose restart

# View resource usage
docker stats --no-stream
```
