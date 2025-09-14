"""
Configuration Management Service

Handles dynamic configuration updates, environment management, and service orchestration.
"""

import os
import json
import yaml
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import uuid
from pathlib import Path

from backend.models.model_config import ModelConfig


@dataclass
class Configuration:
    """A configuration setting."""
    key: str
    value: Any
    category: str  # service, model, system, feature
    description: str
    data_type: str  # string, int, float, bool, json
    default_value: Any
    is_required: bool = True
    is_sensitive: bool = False
    last_updated: str = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()


@dataclass
class ServiceConfig:
    """Service configuration."""
    service_name: str
    enabled: bool
    port: int
    environment_vars: Dict[str, str]
    command: List[str]
    restart_policy: str = "always"
    health_check_url: Optional[str] = None


@dataclass
class ConfigurationChange:
    """A configuration change record."""
    change_id: str
    timestamp: str
    user: str
    service: str
    key: str
    old_value: Any
    new_value: Any
    change_type: str  # create, update, delete
    reason: str = ""


class ConfigManager:
    """Manages system configuration and dynamic updates."""
    
    def __init__(self, config_dir: str = "/app/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.configurations: Dict[str, Configuration] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.change_history: List[ConfigurationChange] = []
        
        # Initialize default configurations
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default system configurations."""
        default_configs = [
            {
                "key": "ollama_base_url",
                "value": "http://host.docker.internal:11434",
                "category": "service",
                "description": "Ollama API base URL",
                "data_type": "string",
                "default_value": "http://host.docker.internal:11434"
            },
            {
                "key": "max_iterations",
                "value": 50,
                "category": "ab_mcts",
                "description": "Maximum AB-MCTS iterations",
                "data_type": "int",
                "default_value": 50
            },
            {
                "key": "max_depth",
                "value": 10,
                "category": "ab_mcts",
                "description": "Maximum search depth",
                "data_type": "int",
                "default_value": 10
            },
            {
                "key": "log_level",
                "value": "info",
                "category": "system",
                "description": "Logging level",
                "data_type": "string",
                "default_value": "info"
            },
            {
                "key": "enable_websockets",
                "value": True,
                "category": "feature",
                "description": "Enable WebSocket connections",
                "data_type": "bool",
                "default_value": True
            },
            {
                "key": "rate_limit_per_minute",
                "value": 100,
                "category": "service",
                "description": "API rate limit per minute",
                "data_type": "int",
                "default_value": 100
            }
        ]
        
        for config_data in default_configs:
            config = Configuration(**config_data)
            self.configurations[config.key] = config
    
    def get_config(self, key: str) -> Optional[Configuration]:
        """Get a configuration by key."""
        return self.configurations.get(key)
    
    def get_configs_by_category(self, category: str) -> List[Configuration]:
        """Get all configurations in a category."""
        return [config for config in self.configurations.values() if config.category == category]
    
    def set_config(self, key: str, value: Any, user: str = "system", reason: str = "") -> bool:
        """Set a configuration value."""
        if key not in self.configurations:
            # Create new configuration
            config = Configuration(
                key=key,
                value=value,
                category="custom",
                description="User-defined configuration",
                data_type=type(value).__name__,
                default_value=value
            )
            self.configurations[key] = config
            change_type = "create"
            old_value = None
        else:
            # Update existing configuration
            config = self.configurations[key]
            old_value = config.value
            config.value = value
            config.last_updated = datetime.now().isoformat()
            change_type = "update"
        
        # Record change
        change = ConfigurationChange(
            change_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            user=user,
            service="config_manager",
            key=key,
            old_value=old_value,
            new_value=value,
            change_type=change_type,
            reason=reason
        )
        self.change_history.append(change)
        
        # Apply configuration if needed
        self._apply_configuration(key, value)
        
        return True
    
    def _apply_configuration(self, key: str, value: Any):
        """Apply a configuration change."""
        # Set environment variable
        os.environ[key.upper()] = str(value)
        
        # Special handling for specific configurations
        if key == "log_level":
            # Update logging configuration
            import logging
            logging.getLogger().setLevel(getattr(logging, value.upper()))
        
        elif key == "ollama_base_url":
            # Update service configurations
            for service_config in self.service_configs.values():
                if "OLLAMA_BASE_URL" in service_config.environment_vars:
                    service_config.environment_vars["OLLAMA_BASE_URL"] = value
    
    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get service configuration."""
        return self.service_configs.get(service_name)
    
    def update_service_config(self, service_name: str, config_data: Dict[str, Any]) -> bool:
        """Update service configuration."""
        if service_name in self.service_configs:
            # Update existing
            service_config = self.service_configs[service_name]
            for key, value in config_data.items():
                if hasattr(service_config, key):
                    setattr(service_config, key, value)
        else:
            # Create new
            service_config = ServiceConfig(service_name=service_name, **config_data)
            self.service_configs[service_name] = service_config
        
        return True
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a service."""
        try:
            # In production, this would use Docker Compose or Kubernetes
            # For now, we'll simulate it
            print(f"Restarting service: {service_name}")
            
            # Record the restart
            change = ConfigurationChange(
                change_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                user="system",
                service=service_name,
                key="restart",
                old_value="running",
                new_value="restarted",
                change_type="update",
                reason="Service restart requested"
            )
            self.change_history.append(change)
            
            return True
        except Exception as e:
            print(f"Failed to restart service {service_name}: {e}")
            return False
    
    async def restart_all_services(self) -> Dict[str, bool]:
        """Restart all services."""
        results = {}
        
        for service_name in self.service_configs:
            results[service_name] = await self.restart_service(service_name)
        
        return results
    
    def export_configuration(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Export configuration for backup."""
        configs_to_export = self.configurations
        
        if category:
            configs_to_export = {
                k: v for k, v in configs_to_export.items() 
                if v.category == category
            }
        
        return {
            "export_timestamp": datetime.now().isoformat(),
            "category": category,
            "configurations": {
                k: asdict(v) for k, v in configs_to_export.items()
            },
            "service_configs": {
                k: asdict(v) for k, v in self.service_configs.items()
            }
        }
    
    def import_configuration(self, config_data: Dict[str, Any]) -> int:
        """Import configuration from backup."""
        imported_count = 0
        
        # Import configurations
        for key, config_dict in config_data.get("configurations", {}).items():
            config = Configuration(**config_dict)
            self.configurations[key] = config
            imported_count += 1
        
        # Import service configurations
        for service_name, service_dict in config_data.get("service_configs", {}).items():
            service_config = ServiceConfig(**service_dict)
            self.service_configs[service_name] = service_config
            imported_count += 1
        
        return imported_count
    
    def get_change_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return [asdict(change) for change in self.change_history[-limit:]]
    
    def validate_configuration(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate configuration data."""
        errors = []
        
        for key, value in config_data.items():
            if key in self.configurations:
                config = self.configurations[key]
                
                # Type validation
                expected_type = config.data_type
                if expected_type == "int" and not isinstance(value, int):
                    errors.append(f"{key} should be an integer")
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"{key} should be a number")
                elif expected_type == "bool" and not isinstance(value, bool):
                    errors.append(f"{key} should be a boolean")
                elif expected_type == "string" and not isinstance(value, str):
                    errors.append(f"{key} should be a string")
        
        return errors
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        total_configs = len(self.configurations)
        total_services = len(self.service_configs)
        
        # Check service health (simplified)
        service_status = {}
        for service_name, service_config in self.service_configs.items():
            service_status[service_name] = {
                "enabled": service_config.enabled,
                "port": service_config.port,
                "health_check": service_config.health_check_url
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_configurations": total_configs,
            "total_services": total_services,
            "service_status": service_status,
            "recent_changes": len([
                change for change in self.change_history
                if datetime.fromisoformat(change.timestamp) > 
                datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            ])
        }
    
    def create_backup(self) -> str:
        """Create a configuration backup."""
        backup_data = self.export_configuration()
        backup_filename = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = self.config_dir / backup_filename
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        return str(backup_path)
    
    def restore_backup(self, backup_path: str) -> int:
        """Restore configuration from backup."""
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        
        return self.import_configuration(backup_data)
