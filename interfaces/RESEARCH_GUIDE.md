# Research Guide: AB-MCTS & Multi-Model

## Overview
This guide summarizes how to collect, visualize, and analyze research signals from AB‑MCTS and Multi‑Model, aligning with goals inspired by Sakana AI’s AB‑MCTS work.

Key layers:
- Run-level artifacts: ExperimentLogger (SQLite index + JSONL events)
- Time-series metrics: Prometheus (/metrics) + Grafana dashboard
- UI research views: interfaces/evals.html (charts, D3 tree, compare runs)

## Prerequisites
- docker-compose up -d
- Evals: http://localhost:8081/evals.html
- Grafana: http://localhost:3001 (Prometheus datasource provisioned)
- Prometheus: http://localhost:9090
- Backend API docs: http://localhost:8095/api/docs

## Data Sources
- Per-run (ExperimentLogger):
  - Index: logs/runs.db (SQLite)
  - Events: logs/runs/YYYYMMDD/run_<id>.jsonl
  - API: /api/runs, /api/runs/{run_id}, /api/runs/{run_id}/events
- Aggregated:
  - /api/monitoring/performance (per-service rates and latencies)
  - /api/monitoring/passk?k=K&hours=H (simple Pass@k over recent runs)
- Metrics (Prometheus):
  - backend_http_requests_total, backend_http_request_latency_seconds
  - ab_mcts_queries_total, ab_mcts_success_total, ab_mcts_latency_seconds
  - ab_mcts_iterations, ab_mcts_nodes_created
  - multi_model_queries_total, multi_model_success_total, multi_model_latency_seconds
  - multi_model_model_usage_total{model="..."}

## Evals & Visualizations (interfaces/evals.html)
- Recent Runs: pick a run to render charts + D3 tree
- Charts:
  - Model Usage (bar): search_stats.model_usage
  - Best Quality by Iteration (line): computed from iteration_log
  - Width vs Depth (stacked bar): search_stats.width_searches/depth_searches
- D3 iteration tree:
  - Layered DAG; node fill by model; stroke by search type
  - Hover for model/quality; text fallback is preserved below
- Compare Runs:
  - Enter comma-separated run IDs
  - Overlays best-quality-by-iteration across runs
  - Stacked model usage across runs

## Grafana (starter dashboard)
- Panels include:
  - AB‑MCTS latency p95; Multi‑Model latency p95
  - Backend HTTP p95 by path
  - Multi‑Model model usage rate by model
  - AB‑MCTS success rate (5m window)
- Suggested additions:
  - AB‑MCTS iterations and nodes distributions
  - Per-model success/latency breakdowns
  - Alerts: p95 latency above thresholds; success rate below thresholds

## Suggested Studies
1) Budget vs Success (Pass@k):
- Use /api/monitoring/passk with different k, H values
- Compare across days or model selections

2) Exploration Patterns:
- Compare width/depth counts and best-quality trajectories across runs
- Identify when deeper refinement pays off vs wider exploration

3) Model Allocation:
- Track model usage distributions by task types
- Identify preferred models for successful runs

4) Latency & Reliability:
- Monitor AB‑MCTS/Multi‑Model latency distributions and success rates
- Correlate with model choices and parameters (iterations, depth)

## Tips
- For reproducibility, snapshot run JSONL and config (MODEL_INTEGRATION config, model lists)
- Keep notes linking run IDs to experiment tags (prompt category, constraints)
- Use Grafana dashboard links for sharing time-windowed views with teammates

## References
- Sakana AI AB‑MCTS summary: https://sakana.ai/ab-mcts/
- Algorithm Code: https://github.com/SakanaAI/treequest
- ARC-AGI-2 Experiments: https://github.com/SakanaAI/ab-mcts-arc2

