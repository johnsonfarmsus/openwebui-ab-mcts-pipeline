"""
Experiment/run logger that stores structured metadata in SQLite and streams
append-only JSONL artifacts to disk. Designed for lightweight, reproducible
research logging without external dependencies.
"""

from __future__ import annotations

import os
import json
import uuid
import time
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


class ExperimentLogger:
    """Minimal run logger with SQLite index and JSONL artifacts.

    Directory layout:
      logs/
        runs.db                  # SQLite index
        runs/
          YYYYMMDD/
            run_<run_id>.jsonl  # line-delimited JSON events
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = base_dir or os.getenv("LOGS_DIR", os.path.abspath("logs"))
        self.runs_dir = os.path.join(self.base_dir, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)

        self.db_path = os.path.join(self.base_dir, "runs.db")
        self._init_db()

    # ---------- Public API ----------
    def start_run(
        self,
        pipeline: str,
        user_query: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        started_at = time.time()
        day_dir = os.path.join(self.runs_dir, time.strftime("%Y%m%d", time.localtime(started_at)))
        os.makedirs(day_dir, exist_ok=True)
        artifact_path = os.path.join(day_dir, f"run_{run_id}.jsonl")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, pipeline, user_query, parameters_json, status, started_at, artifact_path, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    pipeline,
                    user_query,
                    json.dumps(parameters or {}),
                    "running",
                    started_at,
                    artifact_path,
                    json.dumps(metadata or {}),
                ),
            )

        # First line in artifact
        self._write_event(artifact_path, {
            "type": "run_started",
            "ts": started_at,
            "run_id": run_id,
            "pipeline": pipeline,
            "parameters": parameters,
            "metadata": metadata or {},
        })
        return run_id

    def log_event(self, run_id: str, event: Dict[str, Any]) -> None:
        artifact_path = self._get_artifact_path(run_id)
        if not artifact_path:
            return
        event.setdefault("ts", time.time())
        event.setdefault("run_id", run_id)
        self._write_event(artifact_path, event)

    def finish_run(self, run_id: str, result: Dict[str, Any]) -> None:
        finished_at = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, result_json = ? WHERE run_id = ?",
                ("completed", finished_at, json.dumps(result or {}), run_id),
            )

        artifact_path = self._get_artifact_path(run_id)
        if artifact_path:
            self._write_event(artifact_path, {
                "type": "run_finished",
                "ts": finished_at,
                "run_id": run_id,
                "result": result or {},
            })

    def fail_run(self, run_id: str, error: str) -> None:
        finished_at = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE run_id = ?",
                ("failed", finished_at, error, run_id),
            )
        artifact_path = self._get_artifact_path(run_id)
        if artifact_path:
            self._write_event(artifact_path, {
                "type": "run_failed",
                "ts": finished_at,
                "run_id": run_id,
                "error": error,
            })

    def list_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, pipeline, status, started_at, finished_at FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": r[0],
                "pipeline": r[1],
                "status": r[2],
                "started_at": r[3],
                "finished_at": r[4],
            }
            for r in rows
        ]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id, pipeline, user_query, parameters_json, status, started_at, finished_at, artifact_path, result_json, error, metadata_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "run_id": row[0],
            "pipeline": row[1],
            "user_query": row[2],
            "parameters": json.loads(row[3] or "{}"),
            "status": row[4],
            "started_at": row[5],
            "finished_at": row[6],
            "artifact_path": row[7],
            "result": json.loads(row[8] or "{}"),
            "error": row[9],
            "metadata": json.loads(row[10] or "{}"),
        }

    def read_events(self, run_id: str, head: Optional[int] = None) -> List[Dict[str, Any]]:
        artifact_path = self._get_artifact_path(run_id)
        if not artifact_path or not os.path.exists(artifact_path):
            return []
        events: List[Dict[str, Any]] = []
        with open(artifact_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if head is not None and (i + 1) >= head:
                    break
        return events

    # ---------- Internal ----------
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  pipeline TEXT NOT NULL,
                  user_query TEXT,
                  parameters_json TEXT,
                  status TEXT,
                  started_at REAL,
                  finished_at REAL,
                  artifact_path TEXT,
                  result_json TEXT,
                  error TEXT,
                  metadata_json TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        return conn

    def _get_artifact_path(self, run_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT artifact_path FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def _write_event(path: str, event: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


