"""
Scientific Enrichment API

Provides endpoints for chemistry and materials science data enrichment:
- PubChem: Chemical structure and property data
- ChEMBL: Bioactivity, drug targets, and clinical information
- Materials Project: Computational materials properties
- RDKit: Molecular descriptor calculations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import httpx
import uvicorn
import os
import json
from urllib.parse import quote

app = FastAPI(
    title="Scientific Enrichment API",
    version="1.0.0",
    description="Chemistry and materials science data enrichment tools"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MATERIALS_PROJECT_API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY", "")
MP_API_KEY = os.getenv("MP_API_KEY", MATERIALS_PROJECT_API_KEY)
MP_PROXY_URL = os.getenv("MP_PROXY_URL", "")

# Check for optional dependencies
try:
    from mp_api.client import MPRester  # type: ignore
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    from rdkit.Chem import rdMolDescriptors
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Scientific Enrichment API", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "scientific-enrichment",
        "rdkit_available": RDKit_AVAILABLE,
        "materials_project_available": bool(MATERIALS_PROJECT_API_KEY),
    }

@app.get("/tools")
async def list_tools():
    """List available tools."""
    return {
        "tools": [
            {"name": "chem_lipinski_pains", "description": "Check SMILES for Lipinski rule-of-five and PAINS alerts", "endpoint": "/tools/chem/lipinski_pains"},
            {"name": "materials_project_lookup", "description": "Lookup materials by formula, mp-id, or elements using the Materials Project API", "endpoint": "/tools/materials/lookup"},
            {"name": "pubchem_lookup", "description": "Resolve chemical name via PubChem and return short description/properties", "endpoint": "/tools/pubchem/lookup"},
            {"name": "chembl_lookup", "description": "Lookup bioactivity, targets, and drug information from ChEMBL", "endpoint": "/tools/chembl/lookup"},
        ]
    }

# --- Chemistry tools (RDKit-based, graceful fallback) ---
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
    """Lookup by formula, mp-id, or elements via Materials Project v2 API (if key set).

    Request: {"formula": "LiFePO4"} or {"mp_id": "mp-149"} or {"elements": "Pm"} or {"elements": ["Li","Fe","O"]}
    """
    if not MATERIALS_PROJECT_API_KEY:
        return {"success": False, "error": "MATERIALS_PROJECT_API_KEY not set"}
    base = "https://api.materialsproject.org"
    base_v2 = f"{base}/v2"
    headers = {"accept": "application/json", "X-API-KEY": MATERIALS_PROJECT_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=20.0, headers=headers, follow_redirects=True) as client:
            # Rich set of commonly used materials fields
            fields = (
                "material_id,formula_pretty,energy_above_hull,e_above_hull,formation_energy_per_atom,"
                "band_gap,is_metal,density,nelements,spacegroup,spacegroup_symbol,"
                "volume,elements,elasticity,oxidation_states,structure"
            )
            fields_list = [f.strip() for f in fields.split(",") if f.strip()]
            # Prefer GraphQL or official client for richer field selection when available
            if MP_PROXY_URL:
                # Prefer sidecar proxy using official client for robust fields
                try:
                    payload_out: Dict[str, Any] = {k: payload.get(k) for k in ("formula", "mp_id", "elements") if payload.get(k) is not None}
                    payload_out["limit"] = 3
                    pr = await client.post(f"{MP_PROXY_URL}/lookup", json=payload_out)
                    if pr.status_code == 200 and pr.json().get("success"):
                        return {"success": True, "data": pr.json().get("data", [])}
                except Exception:
                    pass
            if MP_API_KEY:
                try:
                    if payload.get("mp_id"):
                        mp_id = payload["mp_id"]
                        gql = {
                            "query": (
                                "{ materialsSummary(material_ids: [\"" + mp_id + "\"], limit: 5) "
                                "{ material_id formula_pretty band_gap is_metal density spacegroup { symbol } e_above_hull formation_energy_per_atom } }"
                            )
                        }
                    elif payload.get("formula"):
                        formula = payload["formula"]
                        gql = {
                            "query": (
                                "{ materialsSummary(formula: \"" + formula + "\", limit: 5) "
                                "{ material_id formula_pretty band_gap is_metal density spacegroup { symbol } e_above_hull formation_energy_per_atom } }"
                            )
                        }
                    else:
                        gql = None
                    if gql:
                        gr = await client.post(f"{base}/graphql", json=gql)
                        if gr.status_code == 200:
                            gj = gr.json()
                            if gj.get("data") and gj["data"].get("materialsSummary") is not None:
                                data = gj["data"]["materialsSummary"]
                                # Limit to top 3 concise rows
                                return {"success": True, "data": data[:3]}
                except Exception:
                    pass

            if _MP_AVAILABLE and MP_API_KEY:
                # Use official client for robust querying
                try:
                    with MPRester(MP_API_KEY) as mpr:
                        if payload.get("mp_id"):
                            docs = mpr.materials.summary.search(material_ids=[payload["mp_id"]], fields=[
                                "material_id","formula_pretty","e_above_hull","formation_energy_per_atom",
                                "band_gap","is_metal","density","nelements","spacegroup.symbol",
                            ])
                        elif payload.get("formula"):
                            docs = mpr.materials.summary.search(formula=payload["formula"], fields=[
                                "material_id","formula_pretty","e_above_hull","formation_energy_per_atom",
                                "band_gap","is_metal","density","nelements","spacegroup.symbol",
                            ],
                            chunk_size=5)
                        elif payload.get("elements"):
                            els = payload["elements"]
                            if isinstance(els, list):
                                docs = mpr.materials.summary.search(elements=els, fields=[
                                    "material_id","formula_pretty","band_gap","density","spacegroup.symbol",
                                ], chunk_size=5)
                            else:
                                docs = mpr.materials.summary.search(elements=[str(els)], fields=[
                                    "material_id","formula_pretty","band_gap","density","spacegroup.symbol",
                                ], chunk_size=5)
                        else:
                            raise HTTPException(status_code=400, detail="Provide 'formula' or 'mp_id'")
                    # The client returns pydantic models; convert to dicts
                    def _to_dict(d):
                        try:
                            return d.model_dump()
                        except Exception:
                            return json.loads(json.dumps(d, default=str))
                    return {"success": True, "data": [_to_dict(d) for d in list(docs)[:5]]}
                except Exception as e:
                    # Fallback to HTTP below
                    pass

            if payload.get("mp_id"):
                mp_id = payload["mp_id"]
                # Use summary fields for richer content
                r = await client.get(f"{base_v2}/materials/summary/{mp_id}?fields={fields}")
                if r.status_code == 404:
                    # Fallback to generic materials endpoint
                    r = await client.get(f"{base_v2}/materials/{mp_id}")
                r.raise_for_status()
                return {"success": True, "data": r.json()}
            elif payload.get("formula"):
                formula = payload["formula"]
                # Strategy: fetch top IDs via REST, then query details via GraphQL by IDs
                # 1) Get IDs
                ids_resp = await client.get(f"{base}/materials/summary/?formula={formula}")
                ids_resp.raise_for_status()
                ids_json = ids_resp.json()
                id_list = [d.get("material_id") for d in (ids_json.get("data") or []) if d.get("material_id")] if isinstance(ids_json, dict) else []
                id_list = id_list[:5]
                if MP_API_KEY and id_list:
                    # 2) GraphQL details by IDs
                    try:
                        gql = {
                            "query": (
                                "{ materialsSummary(material_ids: [" + ",".join(["\\\""+i+"\\\"" for i in id_list]) + "] )"
                                " { material_id formula_pretty band_gap is_metal density spacegroup { symbol } e_above_hull formation_energy_per_atom } }"
                            )
                        }
                        gr = await client.post(f"{base}/graphql", json=gql)
                        if gr.status_code == 200 and gr.json().get("data"):
                            return {"success": True, "data": gr.json()["data"]["materialsSummary"][:3]}
                    except Exception:
                        pass
                # 3) Fallback: hydrate minimal fields via v2 core/?material_ids=
                if id_list:
                    # Try batch core hydration
                    got_details: Optional[List[Dict[str, Any]]] = None
                    try:
                        ids_param = ",".join(id_list)
                        core_r = await client.get(f"{base}/materials/core/?material_ids={ids_param}")
                        if core_r.status_code == 200:
                            core_j = core_r.json()
                            core_map = {}
                            if isinstance(core_j, dict) and isinstance(core_j.get("data"), list):
                                for row in core_j["data"]:
                                    mid = row.get("material_id")
                                    if mid:
                                        core_map[mid] = {
                                            "material_id": mid,
                                            "formula_pretty": row.get("formula_pretty"),
                                            "band_gap": row.get("band_gap"),
                                            "is_metal": row.get("is_metal"),
                                            "density": row.get("density"),
                                            "spacegroup_symbol": (row.get("spacegroup") or {}).get("symbol") if isinstance(row.get("spacegroup"), dict) else None,
                                        }
                                got_details = [
                                    {
                                        **(core_map.get(mid, {"material_id": mid})),
                                        "url": f"https://materialsproject.org/materials/{mid}",
                                    }
                                    for mid in id_list
                                ]
                    except Exception:
                        got_details = None
                    # Sequential per-id fallback if batch fails
                    if not got_details:
                        seq: List[Dict[str, Any]] = []
                        for mid in id_list:
                            try:
                                rr = await client.get(f"{base}/materials/core/?material_ids={mid}")
                                if rr.status_code == 200:
                                    rj = rr.json()
                                    rows = (rj.get("data") if isinstance(rj, dict) else None) or []
                                    if rows:
                                        row = rows[0]
                                        seq.append({
                                            "material_id": mid,
                                            "formula_pretty": row.get("formula_pretty"),
                                            "band_gap": row.get("band_gap"),
                                            "is_metal": row.get("is_metal"),
                                            "density": row.get("density"),
                                            "spacegroup_symbol": (row.get("spacegroup") or {}).get("symbol") if isinstance(row.get("spacegroup"), dict) else None,
                                            "url": f"https://materialsproject.org/materials/{mid}",
                                        })
                                    else:
                                        seq.append({"material_id": mid})
                                else:
                                    seq.append({"material_id": mid})
                            except Exception:
                                seq.append({"material_id": mid})
                        got_details = seq
                    # Limit to top 3
                    return {"success": True, "data": got_details[:3]}
                # Fallback: return shallow IDs payload
                return {"success": True, "data": (ids_json or [])}
            elif payload.get("elements"):
                els = payload["elements"]
                if isinstance(els, list):
                    els_param = "-".join(els)
                else:
                    els_param = str(els)
                # Strategy: fetch IDs via REST, then details via GraphQL by IDs if available
                ids_resp = await client.get(f"{base}/materials/summary/?elements={els_param}")
                ids_resp.raise_for_status()
                ids_json = ids_resp.json()
                id_list = [d.get("material_id") for d in (ids_json.get("data") or []) if d.get("material_id")] if isinstance(ids_json, dict) else []
                id_list = id_list[:5]
                if MP_API_KEY and id_list:
                    try:
                        gql = {
                            "query": (
                                "{ materialsSummary(material_ids: [" + ",".join(["\\\""+i+"\\\"" for i in id_list]) + "] )"
                                " { material_id formula_pretty band_gap is_metal density spacegroup { symbol } } }"
                            )
                        }
                        gr = await client.post(f"{base}/graphql", json=gql)
                        if gr.status_code == 200 and gr.json().get("data"):
                            return {"success": True, "data": gr.json()["data"]["materialsSummary"][:3]}
                    except Exception:
                        pass
                if id_list:
                    got_details: Optional[List[Dict[str, Any]]] = None
                    try:
                        ids_param = ",".join(id_list)
                        core_r = await client.get(f"{base}/materials/core/?material_ids={ids_param}")
                        if core_r.status_code == 200:
                            core_j = core_r.json()
                            core_map = {}
                            if isinstance(core_j, dict) and isinstance(core_j.get("data"), list):
                                for row in core_j["data"]:
                                    mid = row.get("material_id")
                                    if mid:
                                        core_map[mid] = {
                                            "material_id": mid,
                                            "formula_pretty": row.get("formula_pretty"),
                                            "band_gap": row.get("band_gap"),
                                            "is_metal": row.get("is_metal"),
                                            "density": row.get("density"),
                                            "spacegroup_symbol": (row.get("spacegroup") or {}).get("symbol") if isinstance(row.get("spacegroup"), dict) else None,
                                        }
                                got_details = [
                                    {
                                        **(core_map.get(mid, {"material_id": mid})),
                                        "url": f"https://materialsproject.org/materials/{mid}",
                                    }
                                    for mid in id_list
                                ]
                    except Exception:
                        got_details = None
                    if not got_details:
                        seq: List[Dict[str, Any]] = []
                        for mid in id_list:
                            try:
                                rr = await client.get(f"{base}/materials/core/?material_ids={mid}")
                                if rr.status_code == 200:
                                    rj = rr.json()
                                    rows = (rj.get("data") if isinstance(rj, dict) else None) or []
                                    if rows:
                                        row = rows[0]
                                        seq.append({
                                            "material_id": mid,
                                            "formula_pretty": row.get("formula_pretty"),
                                            "band_gap": row.get("band_gap"),
                                            "is_metal": row.get("is_metal"),
                                            "density": row.get("density"),
                                            "spacegroup_symbol": (row.get("spacegroup") or {}).get("symbol") if isinstance(row.get("spacegroup"), dict) else None,
                                            "url": f"https://materialsproject.org/materials/{mid}",
                                        })
                                    else:
                                        seq.append({"material_id": mid})
                                else:
                                    seq.append({"material_id": mid})
                            except Exception:
                                seq.append({"material_id": mid})
                        got_details = seq
                    return {"success": True, "data": got_details[:3]}
                return {"success": True, "data": (ids_json or [])}
            else:
                raise HTTPException(status_code=400, detail="Provide 'formula' or 'mp_id'")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Materials Project HTTP error: {str(e)}")

# --- PubChem lookup (no API key required) ---
@app.post("/tools/pubchem/lookup")
async def pubchem_lookup(payload: Dict[str, Any]):
    """Resolve a chemical name or formula via PubChem and return a concise summary.

    Request: {"name": "promethium"} or {"formula": "LiFePO4"}
    """
    name = (payload or {}).get("name", "").strip()
    formula = (payload or {}).get("formula", "").strip()
    if not name and not formula:
        raise HTTPException(status_code=400, detail="Provide 'name' or 'formula'")
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Resolve to a CID either by name or by formula
            if name:
                r = await client.get(f"{base}/compound/name/{quote(name, safe='')}/cids/JSON")
            else:
                r = await client.get(f"{base}/compound/formula/{quote(formula, safe='')}/cids/JSON")
            r.raise_for_status()
            cids = r.json().get("IdentifierList", {}).get("CID", [])
            if not cids:
                return {"success": False, "error": "not_found"}
            cid = cids[0]
            # Get description (PUG View) and a few properties
            desc_r = await client.get(f"{base}/view/data/compound/{cid}/JSON")
            props_r = await client.get(f"{base}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,InChIKey,CanonicalSMILES,IUPACName/JSON")
            desc = ""
            title = None
            try:
                view = desc_r.json()
                # Extract first non-empty description string we can find
                records = view.get("Record", {}).get("Section", [])
                # Heuristic scan
                def find_text(sections):
                    for sec in sections or []:
                        if "Information" in sec:
                            for info in sec["Information"]:
                                val = (info.get("Value", {}) or {}).get("StringWithMarkup", [])
                                if val:
                                    txt = " ".join([v.get("String", "") for v in val]).strip()
                                    if txt:
                                        return txt
                        inner = sec.get("Section")
                        txt = find_text(inner)
                        if txt:
                            return txt
                    return ""
                desc = find_text(records) or ""
                # Try to capture a record title if present
                title = view.get("Record", {}).get("RecordTitle")
            except Exception:
                desc = ""
            props = {}
            smiles = None
            iupac = None
            try:
                props_data = props_r.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                props = {k: props_data.get(k) for k in ("MolecularFormula", "MolecularWeight", "InChIKey", "CanonicalSMILES", "IUPACName")}
                # PubChem returns SMILES as either "CanonicalSMILES" or "ConnectivitySMILES"
                smiles = props_data.get("CanonicalSMILES") or props_data.get("ConnectivitySMILES")
                iupac = props_data.get("IUPACName")
            except Exception:
                props = {}
            # Trim description for prompt safety
            if desc and len(desc) > 600:
                desc = desc[:600] + "…"
            return {"success": True, "cid": cid, "description": desc, "properties": props, "smiles": smiles, "title": title, "iupac": iupac}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"PubChem HTTP error: {str(e)}")

# --- ChEMBL lookup (bioactivity, targets, drug info) ---
@app.post("/tools/chembl/lookup")
async def chembl_lookup(payload: Dict[str, Any]):
    """Lookup bioactivity, targets, and drug information from ChEMBL.

    Request: {"name": "aspirin"} or {"smiles": "CC(=O)Oc1ccccc1C(=O)O"} or {"chembl_id": "CHEMBL25"}
    """
    name = (payload or {}).get("name", "").strip()
    smiles = (payload or {}).get("smiles", "").strip()
    chembl_id = (payload or {}).get("chembl_id", "").strip()

    if not name and not smiles and not chembl_id:
        raise HTTPException(status_code=400, detail="Provide 'name', 'smiles', or 'chembl_id'")

    base = "https://www.ebi.ac.uk/chembl/api/data"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Search for molecule
            if chembl_id:
                # Direct lookup by ChEMBL ID
                r = await client.get(f"{base}/molecule/{chembl_id}.json")
            elif smiles:
                # Search by SMILES
                r = await client.get(f"{base}/molecule/search.json?q={quote(smiles, safe='')}")
            else:
                # Search by name
                r = await client.get(f"{base}/molecule/search.json?q={quote(name, safe='')}")

            r.raise_for_status()
            data = r.json()

            # Extract molecule data
            molecules = data.get("molecules", [])
            if not molecules and "molecule_chembl_id" in data:
                # Direct lookup response
                molecules = [data]

            if not molecules:
                return {"success": False, "error": "not_found"}

            # Get the first (most relevant) molecule
            mol = molecules[0]
            chembl_id = mol.get("molecule_chembl_id")

            # Extract key information
            result = {
                "success": True,
                "chembl_id": chembl_id,
                "pref_name": mol.get("pref_name"),
                "max_phase": mol.get("max_phase"),  # Clinical trial phase (4 = approved drug)
                "first_approval": mol.get("first_approval"),
                "molecule_type": mol.get("molecule_type"),
                "therapeutic_flag": mol.get("therapeutic_flag"),
                "dosed_ingredient": mol.get("dosed_ingredient"),
            }

            # Molecule properties
            props = mol.get("molecule_properties", {})
            if props:
                result["properties"] = {
                    "molecular_weight": props.get("full_mwt"),
                    "alogp": props.get("alogp"),
                    "psa": props.get("psa"),
                    "hba": props.get("hba"),
                    "hbd": props.get("hbd"),
                    "num_ro5_violations": props.get("num_ro5_violations"),
                    "aromatic_rings": props.get("aromatic_rings"),
                }

            # ATC classifications (drug categories)
            atc = mol.get("atc_classifications", [])
            if atc:
                result["atc_codes"] = atc[:5]  # Limit to 5

            # Get bioactivity data (targets)
            try:
                bio_r = await client.get(f"{base}/activity.json?molecule_chembl_id={chembl_id}&limit=10")
                if bio_r.status_code == 200:
                    bio_data = bio_r.json()
                    activities = bio_data.get("activities", [])
                    if activities:
                        targets = []
                        seen_targets = set()
                        for act in activities[:10]:
                            target_id = act.get("target_chembl_id")
                            if target_id and target_id not in seen_targets:
                                seen_targets.add(target_id)
                                targets.append({
                                    "target_id": target_id,
                                    "target_type": act.get("target_type"),
                                    "activity_type": act.get("standard_type"),
                                    "value": act.get("standard_value"),
                                    "units": act.get("standard_units"),
                                })
                        if targets:
                            result["bioactivity_targets"] = targets[:5]  # Top 5 targets
            except Exception:
                pass  # Bioactivity is optional

            # Get indications (what it treats)
            try:
                ind_r = await client.get(f"{base}/drug_indication.json?molecule_chembl_id={chembl_id}&limit=5")
                if ind_r.status_code == 200:
                    ind_data = ind_r.json()
                    indications = ind_data.get("drug_indications", [])
                    if indications:
                        result["indications"] = [
                            {
                                "mesh_heading": ind.get("mesh_heading"),
                                "efo_term": ind.get("efo_term"),
                                "max_phase": ind.get("max_phase_for_ind"),
                            }
                            for ind in indications[:5]
                        ]
            except Exception:
                pass  # Indications are optional

            return result

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"ChEMBL HTTP error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8099)
