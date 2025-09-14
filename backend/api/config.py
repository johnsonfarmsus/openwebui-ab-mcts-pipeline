"""
Configuration Management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.services.config_manager import ConfigManager

router = APIRouter()

# Initialize configuration manager
config_manager = ConfigManager()

@router.get("/")
async def get_all_configurations(category: Optional[str] = Query(None, description="Filter by category")):
    """Get all configurations."""
    try:
        if category:
            configs = config_manager.get_configs_by_category(category)
        else:
            configs = list(config_manager.configurations.values())
        
        return {
            "configurations": [
                {
                    "key": config.key,
                    "value": config.value,
                    "category": config.category,
                    "description": config.description,
                    "data_type": config.data_type,
                    "is_required": config.is_required,
                    "is_sensitive": config.is_sensitive,
                    "last_updated": config.last_updated
                }
                for config in configs
            ],
            "total": len(configs),
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configurations: {str(e)}")

@router.get("/{key}")
async def get_configuration(key: str):
    """Get a specific configuration by key."""
    try:
        config = config_manager.get_config(key)
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        return {
            "key": config.key,
            "value": config.value,
            "category": config.category,
            "description": config.description,
            "data_type": config.data_type,
            "default_value": config.default_value,
            "is_required": config.is_required,
            "is_sensitive": config.is_sensitive,
            "last_updated": config.last_updated
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")

@router.put("/{key}")
async def update_configuration(key: str, config_data: Dict[str, Any]):
    """Update a configuration value."""
    try:
        value = config_data["value"]
        user = config_data.get("user", "api")
        reason = config_data.get("reason", "Configuration update")
        
        # Validate the configuration
        validation_errors = config_manager.validate_configuration({key: value})
        if validation_errors:
            raise HTTPException(status_code=400, detail=f"Validation errors: {validation_errors}")
        
        success = config_manager.set_config(key, value, user, reason)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update configuration")
        
        return {
            "message": "Configuration updated successfully",
            "key": key,
            "new_value": value,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@router.get("/categories/list")
async def get_categories():
    """Get all configuration categories."""
    try:
        categories = set(config.category for config in config_manager.configurations.values())
        return {
            "categories": sorted(list(categories)),
            "total": len(categories),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@router.get("/services/status")
async def get_services_status():
    """Get status of all services."""
    try:
        status = config_manager.get_system_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get services status: {str(e)}")

@router.post("/services/{service_name}/restart")
async def restart_service(service_name: str):
    """Restart a specific service."""
    try:
        success = await config_manager.restart_service(service_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to restart service {service_name}")
        
        return {
            "message": f"Service {service_name} restarted successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart service: {str(e)}")

@router.post("/services/restart-all")
async def restart_all_services():
    """Restart all services."""
    try:
        results = await config_manager.restart_all_services()
        return {
            "message": "Service restart initiated",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart services: {str(e)}")

@router.get("/history")
async def get_change_history(limit: int = Query(100, ge=1, le=1000, description="Number of changes to return")):
    """Get configuration change history."""
    try:
        history = config_manager.get_change_history(limit)
        return {
            "changes": history,
            "total": len(history),
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get change history: {str(e)}")

@router.post("/export")
async def export_configuration(category: Optional[str] = Query(None, description="Export specific category")):
    """Export configuration for backup."""
    try:
        config_data = config_manager.export_configuration(category)
        return {
            "message": "Configuration exported successfully",
            "data": config_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export configuration: {str(e)}")

@router.post("/import")
async def import_configuration(config_data: Dict[str, Any]):
    """Import configuration from backup."""
    try:
        imported_count = config_manager.import_configuration(config_data)
        return {
            "message": "Configuration imported successfully",
            "imported_items": imported_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import configuration: {str(e)}")

@router.post("/backup/create")
async def create_backup():
    """Create a configuration backup file."""
    try:
        backup_path = config_manager.create_backup()
        return {
            "message": "Backup created successfully",
            "backup_path": backup_path,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")

@router.post("/backup/restore")
async def restore_backup(backup_data: Dict[str, Any]):
    """Restore configuration from backup."""
    try:
        backup_path = backup_data["backup_path"]
        restored_count = config_manager.restore_backup(backup_path)
        return {
            "message": "Backup restored successfully",
            "restored_items": restored_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore backup: {str(e)}")

@router.get("/validation/{key}")
async def validate_configuration_value(key: str, value: Any):
    """Validate a configuration value."""
    try:
        errors = config_manager.validate_configuration({key: value})
        return {
            "key": key,
            "value": value,
            "is_valid": len(errors) == 0,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate configuration: {str(e)}")

@router.get("/health")
async def get_configuration_health():
    """Get configuration system health."""
    try:
        status = config_manager.get_system_status()
        return {
            "status": "healthy",
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
