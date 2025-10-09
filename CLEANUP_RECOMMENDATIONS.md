# Project Cleanup Recommendations

This document lists files and directories that can be safely removed or archived to clean up the project.

## ✅ Safe to Remove

### Old AB-MCTS Service Iterations
These appear to be development iterations that are no longer used:

**Currently Used (DO NOT DELETE):**
- `backend/services/proper_treequest_ab_mcts_service.py` ✓ (in use)
- `backend/services/proper_multi_model_service.py` ✓ (in use)

**Old Iterations (safe to remove):**
```
backend/services/ab_mcts_service.py
backend/services/conversational_ab_mcts_service.py
backend/services/proper_ab_mcts_service.py
backend/services/real_ab_mcts_service.py
backend/services/sakana_ab_mcts_service.py
backend/services/sakana_treequest_ab_mcts_service.py
backend/services/simple_ab_mcts_service.py
backend/services/simple_multi_model_service.py
backend/services/treequest_ab_mcts_service.py
backend/services/true_sakana_ab_mcts_service.py
```

**Recommendation:** Archive to `backend/services/deprecated/` if you want to keep them for reference.

### Python Cache
```
backend/__pycache__/
**/__pycache__/
*.pyc
*.pyo
```

**Action:** Add to `.gitignore` and remove from repository.

### Test Files
```
functions/test_function.py
sakana-fork/src/ab_mcts_arc2/unittest_templates/test_transform.py
```

**Recommendation:**
- Keep if they're useful for testing
- Move to a dedicated `tests/` directory
- Or remove if no longer needed

### Sakana Fork
```
sakana-fork/
```

**Question:** Is this still needed? If it was used for research but not actively maintained:
- Option 1: Keep as-is (for reference)
- Option 2: Remove and document the Sakana AI dependency in README
- Option 3: Move to a separate research repository

## 📝 Duplicate/Redundant Documentation

**Potential Duplicates to Review:**
```
interfaces/README.md
interfaces/RESEARCH_GUIDE.md
docs/research/RESEARCH_GUIDE.md
```

**Action:** Merge or consolidate if content overlaps.

## 🔧 Recommended .gitignore Additions

Create or update `.gitignore` with:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Logs
*.log
logs/*.db
logs/runs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local
venv/
env/

# OS
.DS_Store
Thumbs.db

# Docker
docker-compose.override.yml

# Jupyter
.ipynb_checkpoints/

# Config persistence (user-specific)
logs/selected_models_*.json
logs/config_*.json
```

## 🗂️ Recommended Directory Structure Cleanup

**Current:**
```
backend/services/
├── [10+ old AB-MCTS files]
├── proper_treequest_ab_mcts_service.py
└── proper_multi_model_service.py
```

**Proposed:**
```
backend/services/
├── ab_mcts_service.py  (rename from proper_treequest_ab_mcts_service.py)
├── multi_model_service.py  (rename from proper_multi_model_service.py)
├── experiment_logger.py
├── model_discovery.py
├── config_persistence.py
└── deprecated/
    └── [old iteration files moved here]
```

## 🎯 Action Plan

1. **Add `.gitignore`** for Python cache and logs
2. **Archive old service files** to `backend/services/deprecated/`
3. **Consider renaming** active services to simpler names
4. **Clean up Python cache** directories
5. **Review and consolidate** duplicate documentation

## ⚠️ Before Deleting

Always verify with:
```bash
# Check if file is imported anywhere
grep -r "from.*filename" .
grep -r "import.*filename" .

# Check docker-compose references
grep -r "filename" docker-compose.yml
```

## 📋 Checklist

- [ ] Create `.gitignore`
- [ ] Remove Python cache directories
- [ ] Archive old AB-MCTS service files
- [ ] Consolidate duplicate docs
- [ ] Update docker-compose if renaming files
- [ ] Test that all services still work
- [ ] Commit changes with descriptive message
