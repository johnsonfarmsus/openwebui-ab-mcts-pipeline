"""
Open WebUI Integration Service

This service provides OpenAPI-compatible endpoints specifically designed for Open WebUI integration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import httpx
import uvicorn
from pydantic import BaseModel
import os

app = FastAPI(
    title="AB-MCTS & Multi-Model Tools",
    version="1.0.0",
    description="Advanced AI tools for AB-MCTS and Multi-Model collaboration"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
AB_MCTS_SERVICE_URL = "http://ab-mcts-service:8094"
MULTI_MODEL_SERVICE_URL = "http://multi-model-service:8090"

# Optional external tools configuration
MATERIALS_PROJECT_API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY", "")

class QueryRequest(BaseModel):
    query: str
    iterations: Optional[int] = 20
    max_depth: Optional[int] = 5
    models: Optional[List[str]] = None

class MultiModelRequest(BaseModel):
    query: str
    models: Optional[List[str]] = None

class ModelUpdateRequest(BaseModel):
    models: List[str]

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Open WebUI Integration Service", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "openwebui-integration"}

# AB-MCTS Endpoints
@app.post("/ab-mcts/query")
async def ab_mcts_query(request: QueryRequest):
    """
    Run AB-MCTS (Adaptive Branching Monte Carlo Tree Search) query.
    
    This tool uses advanced tree search algorithms to solve complex problems
    by exploring multiple solution paths and finding the best answer.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "query": request.query,
                "iterations": request.iterations,
                "max_depth": request.max_depth
            }
            if request.models:
                payload["models"] = request.models
                
            response = await client.post(
                f"{AB_MCTS_SERVICE_URL}/query",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AB-MCTS service error: {str(e)}")

@app.get("/ab-mcts/models")
async def get_ab_mcts_models():
    """Get available AB-MCTS models."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{AB_MCTS_SERVICE_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AB-MCTS models: {str(e)}")

@app.post("/ab-mcts/models/update")
async def update_ab_mcts_models(request: ModelUpdateRequest):
    """Update AB-MCTS model selection."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{AB_MCTS_SERVICE_URL}/models/update",
                json={"models": request.models}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update AB-MCTS models: {str(e)}")

# Multi-Model Endpoints
@app.post("/multi-model/query")
async def multi_model_query(request: MultiModelRequest):
    """
    Run Multi-Model collaboration query.
    
    This tool uses multiple AI models working together to provide
    comprehensive and well-rounded answers to complex questions.
    """
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            payload = {"query": request.query}
            if request.models:
                payload["models"] = request.models
                
            response = await client.post(
                f"{MULTI_MODEL_SERVICE_URL}/query",
                json=payload
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-Model service error: {str(e)}")

@app.get("/multi-model/models")
async def get_multi_model_models():
    """Get available Multi-Model models."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{MULTI_MODEL_SERVICE_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Multi-Model models: {str(e)}")

@app.post("/multi-model/models/update")
async def update_multi_model_models(request: ModelUpdateRequest):
    """Update Multi-Model model selection."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{MULTI_MODEL_SERVICE_URL}/models/update",
                json={"models": request.models}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update Multi-Model models: {str(e)}")

# Explicit tool endpoints for Open WebUI
@app.post("/tools/ab_mcts")
async def tool_ab_mcts(request: QueryRequest):
    """AB-MCTS Tool - Advanced tree search for complex problem solving."""
    return await ab_mcts_query(request)

@app.post("/tools/multi_model")
async def tool_multi_model(request: MultiModelRequest):
    """Multi-Model Tool - Collaborative AI for comprehensive answers."""
    return await multi_model_query(request)

@app.get("/tools")
async def list_tools():
    """List available tools."""
    return {
        "tools": [
            {"name": "ab_mcts", "description": "Advanced tree search for complex problem solving", "endpoint": "/tools/ab_mcts"},
            {"name": "multi_model", "description": "Collaborative AI for comprehensive answers", "endpoint": "/tools/multi_model"},
            {"name": "chem_lipinski_pains", "description": "Check SMILES for Lipinski rule-of-five and PAINS alerts", "endpoint": "/tools/chem/lipinski_pains"},
            {"name": "materials_project_lookup", "description": "Lookup materials by formula or mp-id using the Materials Project API", "endpoint": "/tools/materials/lookup"},
        ]
    }

# --- Chemistry tools (RDKit-based, graceful fallback) ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    from rdkit.Chem import rdMolDescriptors
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

@app.post("/tools/chem/lipinski_pains")
async def chem_lipinski_pains(payload: Dict[str, Any]):
    """Evaluate SMILES for Lipinski rules and (placeholder) PAINS flags.

    Request: {"smiles": "CCO..."}
    """
    smiles = payload.get("smiles", "")
    if not smiles:
        raise HTTPException(status_code=400, detail="Missing 'smiles'")
    if not RDKit_AVAILABLE:
        return {"success": False, "error": "RDKit not installed in this image"}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"success": False, "error": "Invalid SMILES"}
        mw = Descriptors.MolWt(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        logp = Descriptors.MolLogP(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        frac_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
        heavy_atoms = mol.GetNumHeavyAtoms()
        formula = rdMolDescriptors.CalcMolFormula(mol)
        lipinski_pass = (
            (mw <= 500)
            and (hbd <= 5)
            and (hba <= 10)
            and (logp <= 5)
        )
        # Placeholder PAINS: a real implementation would use substructure SMARTS
        pains_alerts: List[str] = []
        return {
            "success": True,
            "lipinski": {
                "mw": mw,
                "hbd": hbd,
                "hba": hba,
                "logp": logp,
                "rotatable_bonds": rot_bonds,
                "tpsa": tpsa,
                "ring_count": rings,
                "fraction_csp3": frac_csp3,
                "heavy_atoms": heavy_atoms,
                "formula": formula,
                "passes": lipinski_pass,
            },
            "pains_alerts": pains_alerts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chem tool error: {str(e)}")

# --- Materials Project lookup ---
@app.post("/tools/materials/lookup")
async def materials_lookup(payload: Dict[str, Any]):
    """Lookup by formula or mp-id via Materials Project v2 API (if key set).

    Request: {"formula": "LiFePO4"} or {"mp_id": "mp-149"}
    """
    if not MATERIALS_PROJECT_API_KEY:
        return {"success": False, "error": "MATERIALS_PROJECT_API_KEY not set"}
    base = "https://api.materialsproject.org"
    headers = {"accept": "application/json", "X-API-KEY": MATERIALS_PROJECT_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
            # Rich set of commonly used materials fields
            fields = (
                "material_id,formula_pretty,energy_above_hull,e_above_hull,formation_energy_per_atom,"
                "band_gap,is_metal,density,nelements,spacegroup,spacegroup_symbol,"
                "volume,elements,elasticity,oxidation_states,structure"
            )
            if payload.get("mp_id"):
                mp_id = payload["mp_id"]
                # Use summary fields for richer content
                r = await client.get(f"{base}/materials/summary/{mp_id}?fields={fields}")
                if r.status_code == 404:
                    # Fallback to generic materials endpoint
                    r = await client.get(f"{base}/materials/{mp_id}")
                r.raise_for_status()
                return {"success": True, "data": r.json()}
            elif payload.get("formula"):
                formula = payload["formula"]
                r = await client.get(f"{base}/materials/summary/?formula={formula}&fields={fields}")
                r.raise_for_status()
                return {"success": True, "data": r.json()}
            else:
                raise HTTPException(status_code=400, detail="Provide 'formula' or 'mp_id'")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Materials Project HTTP error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8097)
