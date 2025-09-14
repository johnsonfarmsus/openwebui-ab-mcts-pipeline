"""
Real-time Monitoring Service

Handles real-time performance monitoring, metrics collection, and WebSocket connections.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import uuid
from collections import defaultdict, deque
import statistics

from backend.models.search_stats import SearchStats
from backend.models.model_config import ModelConfig


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: str
    metric_name: str
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """An alert condition."""
    alert_id: str
    name: str
    condition: str
    threshold: float
    severity: str  # low, medium, high, critical
    enabled: bool = True
    last_triggered: Optional[str] = None


@dataclass
class WebSocketConnection:
    """WebSocket connection info."""
    connection_id: str
    client_type: str  # dashboard, mobile, api
    subscribed_metrics: Set[str]
    last_activity: str


class MonitoringService:
    """Real-time monitoring and metrics collection service."""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Alert] = {}
        self.websocket_connections: Dict[str, WebSocketConnection] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Initialize default alerts
        self._initialize_default_alerts()
    
    def _initialize_default_alerts(self):
        """Initialize default alert conditions."""
        default_alerts = [
            {
                "alert_id": "high_error_rate",
                "name": "High Error Rate",
                "condition": "error_rate > 0.1",
                "threshold": 0.1,
                "severity": "high"
            },
            {
                "alert_id": "slow_response",
                "name": "Slow Response Time",
                "condition": "avg_response_time > 10.0",
                "threshold": 10.0,
                "severity": "medium"
            },
            {
                "alert_id": "low_success_rate",
                "name": "Low Success Rate",
                "condition": "success_rate < 0.8",
                "threshold": 0.8,
                "severity": "high"
            }
        ]
        
        for alert_data in default_alerts:
            alert = Alert(**alert_data)
            self.alerts[alert.alert_id] = alert
    
    def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value."""
        metric_point = MetricPoint(
            timestamp=datetime.now().isoformat(),
            metric_name=metric_name,
            value=value,
            tags=tags or {}
        )
        
        self.metrics[metric_name].append(metric_point)
        
        # Check alerts
        self._check_alerts(metric_name, value, tags)
        
        # Broadcast to WebSocket connections
        self._broadcast_metric(metric_point)
    
    def record_search_stats(self, search_stats: SearchStats, service_name: str):
        """Record search statistics as metrics."""
        timestamp = datetime.now().isoformat()
        tags = {"service": service_name}
        
        # Record various metrics
        self.record_metric("search_iterations", search_stats.total_iterations, tags)
        self.record_metric("nodes_created", search_stats.nodes_created, tags)
        self.record_metric("best_reward", search_stats.best_reward, tags)
        self.record_metric("avg_reward", search_stats.average_reward, tags)
        self.record_metric("exploration_ratio", search_stats.exploration_ratio, tags)
        
        if search_stats.response_time:
            self.record_metric("response_time", search_stats.response_time, tags)
        
        # Record model usage
        for model, count in search_stats.model_usage.items():
            self.record_metric("model_usage", count, {**tags, "model": model})
    
    def record_api_call(self, endpoint: str, response_time: float, status_code: int, service: str):
        """Record API call metrics."""
        tags = {
            "endpoint": endpoint,
            "service": service,
            "status_code": str(status_code)
        }
        
        self.record_metric("api_response_time", response_time, tags)
        self.record_metric("api_calls", 1, tags)
        
        if status_code >= 400:
            self.record_metric("api_errors", 1, tags)
    
    def get_metrics(self, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics for a specific metric name."""
        if metric_name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = []
        
        for metric_point in self.metrics[metric_name]:
            metric_time = datetime.fromisoformat(metric_point.timestamp)
            if metric_time >= cutoff_time:
                recent_metrics.append(asdict(metric_point))
        
        return recent_metrics
    
    def get_aggregated_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get aggregated metrics for dashboard."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Aggregate by service
        service_metrics = defaultdict(lambda: {
            "total_calls": 0,
            "total_errors": 0,
            "response_times": [],
            "success_rate": 0.0
        })
        
        # Process API calls
        api_calls = self.get_metrics("api_calls", hours)
        api_errors = self.get_metrics("api_errors", hours)
        response_times = self.get_metrics("api_response_time", hours)
        
        for call in api_calls:
            service = call["tags"].get("service", "unknown")
            service_metrics[service]["total_calls"] += call["value"]
        
        for error in api_errors:
            service = error["tags"].get("service", "unknown")
            service_metrics[service]["total_errors"] += error["value"]
        
        for rt in response_times:
            service = rt["tags"].get("service", "unknown")
            service_metrics[service]["response_times"].append(rt["value"])
        
        # Calculate success rates and average response times
        for service, metrics in service_metrics.items():
            if metrics["total_calls"] > 0:
                metrics["success_rate"] = 1.0 - (metrics["total_errors"] / metrics["total_calls"])
            
            if metrics["response_times"]:
                metrics["avg_response_time"] = statistics.mean(metrics["response_times"])
                metrics["p95_response_time"] = self._percentile(metrics["response_times"], 95)
            else:
                metrics["avg_response_time"] = 0.0
                metrics["p95_response_time"] = 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "period_hours": hours,
            "services": dict(service_metrics)
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _check_alerts(self, metric_name: str, value: float, tags: Dict[str, str]):
        """Check if any alerts should be triggered."""
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            # Simple condition evaluation (in production, use a proper expression evaluator)
            should_trigger = False
            
            if alert.condition == "error_rate > 0.1" and metric_name == "api_errors":
                # Calculate error rate
                total_calls = sum(point.value for point in self.metrics["api_calls"])
                total_errors = sum(point.value for point in self.metrics["api_errors"])
                if total_calls > 0:
                    error_rate = total_errors / total_calls
                    should_trigger = error_rate > alert.threshold
            
            elif alert.condition == "avg_response_time > 10.0" and metric_name == "api_response_time":
                should_trigger = value > alert.threshold
            
            elif alert.condition == "success_rate < 0.8" and metric_name == "api_calls":
                # Calculate success rate
                total_calls = sum(point.value for point in self.metrics["api_calls"])
                total_errors = sum(point.value for point in self.metrics["api_errors"])
                if total_calls > 0:
                    success_rate = 1.0 - (total_errors / total_calls)
                    should_trigger = success_rate < alert.threshold
            
            if should_trigger:
                self._trigger_alert(alert, value, tags)
    
    def _trigger_alert(self, alert: Alert, value: float, tags: Dict[str, str]):
        """Trigger an alert."""
        alert.last_triggered = datetime.now().isoformat()
        
        # In production, this would send notifications (email, Slack, etc.)
        print(f"ALERT TRIGGERED: {alert.name} - Value: {value}, Condition: {alert.condition}")
        
        # Broadcast to WebSocket connections
        self._broadcast_alert(alert, value, tags)
    
    def _broadcast_metric(self, metric_point: MetricPoint):
        """Broadcast metric to WebSocket connections."""
        # In production, this would use actual WebSocket broadcasting
        # For now, we'll just log it
        pass
    
    def _broadcast_alert(self, alert: Alert, value: float, tags: Dict[str, str]):
        """Broadcast alert to WebSocket connections."""
        # In production, this would use actual WebSocket broadcasting
        # For now, we'll just log it
        pass
    
    def add_websocket_connection(self, connection_id: str, client_type: str, 
                               subscribed_metrics: Set[str]) -> WebSocketConnection:
        """Add a new WebSocket connection."""
        connection = WebSocketConnection(
            connection_id=connection_id,
            client_type=client_type,
            subscribed_metrics=subscribed_metrics,
            last_activity=datetime.now().isoformat()
        )
        
        self.websocket_connections[connection_id] = connection
        return connection
    
    def remove_websocket_connection(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id in self.websocket_connections:
            del self.websocket_connections[connection_id]
    
    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get list of active WebSocket connections."""
        return [
            {
                "connection_id": conn.connection_id,
                "client_type": conn.client_type,
                "subscribed_metrics": list(conn.subscribed_metrics),
                "last_activity": conn.last_activity
            }
            for conn in self.websocket_connections.values()
        ]
    
    def create_alert(self, name: str, condition: str, threshold: float, severity: str) -> str:
        """Create a new alert."""
        alert_id = str(uuid.uuid4())
        alert = Alert(
            alert_id=alert_id,
            name=name,
            condition=condition,
            threshold=threshold,
            severity=severity
        )
        
        self.alerts[alert_id] = alert
        return alert_id
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get all alerts."""
        return [asdict(alert) for alert in self.alerts.values()]
    
    def get_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive health dashboard data."""
        aggregated = self.get_aggregated_metrics(hours=1)
        
        # Calculate overall health score
        total_services = len(aggregated["services"])
        healthy_services = len([
            s for s in aggregated["services"].values() 
            if s["success_rate"] > 0.9 and s["avg_response_time"] < 5.0
        ])
        
        health_score = (healthy_services / total_services * 100) if total_services > 0 else 100
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health_score": health_score,
            "total_services": total_services,
            "healthy_services": healthy_services,
            "active_alerts": len([a for a in self.alerts.values() if a.last_triggered]),
            "websocket_connections": len(self.websocket_connections),
            "metrics": aggregated
        }
