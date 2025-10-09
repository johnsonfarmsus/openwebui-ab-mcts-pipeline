"""
Configuration Persistence Service

Saves and loads configuration settings to SQLite database.
Ensures settings persist across container/server restarts.
"""

import sqlite3
import json
import os
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from pathlib import Path


class ConfigPersistence:
    """Manages persistent configuration storage."""

    def __init__(self, db_path: str = "/app/logs/config.db"):
        """Initialize configuration persistence.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    service TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def save_config(self, service: str, config: Dict[str, Any]):
        """Save configuration for a service.

        Args:
            service: Service name (e.g., 'ab-mcts', 'multi-model')
            config: Configuration dictionary
        """
        config_json = json.dumps(config)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO config (service, config_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (service, config_json))

    def load_config(self, service: str) -> Optional[Dict[str, Any]]:
        """Load configuration for a service.

        Args:
            service: Service name

        Returns:
            Configuration dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT config_json FROM config WHERE service = ?",
                (service,)
            )
            row = cursor.fetchone()

            if row:
                return json.loads(row['config_json'])
            return None

    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all service configurations.

        Returns:
            Dictionary mapping service names to configurations
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT service, config_json FROM config")
            return {
                row['service']: json.loads(row['config_json'])
                for row in cursor.fetchall()
            }

    def delete_config(self, service: str):
        """Delete configuration for a service.

        Args:
            service: Service name
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM config WHERE service = ?", (service,))


# Global instance
_config_persistence = None


def get_config_persistence() -> ConfigPersistence:
    """Get global configuration persistence instance."""
    global _config_persistence
    if _config_persistence is None:
        _config_persistence = ConfigPersistence()
    return _config_persistence
