from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

app = FastAPI(title="MP Proxy", version="1.0.0")

MP_API_KEY = os.getenv("MP_API_KEY", os.getenv("MATERIALS_PROJECT_API_KEY", ""))

try:
    from mp_api.client import MPRester  # type: ignore
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False


class LookupPayload(BaseModel):
    formula: Optional[str] = None
    mp_id: Optional[str] = None
    elements: Optional[List[str] | str] = None
    limit: Optional[int] = 3


@app.get("/health")
async def health():
    return {"status": "ok", "mp_available": _MP_AVAILABLE}


@app.post("/lookup")
async def lookup(payload: LookupPayload) -> Dict[str, Any]:
    if not _MP_AVAILABLE or not MP_API_KEY:
        raise HTTPException(status_code=503, detail="MP client unavailable or API key missing")
    try:
        fields = [
            "material_id",
            "formula_pretty",
            "e_above_hull",
            "formation_energy_per_atom",
            "band_gap",
            "is_metal",
            "density",
            "spacegroup.symbol",
        ]
        results: List[Dict[str, Any]] = []
        with MPRester(MP_API_KEY) as mpr:
            if payload.mp_id:
                docs = mpr.materials.summary.search(material_ids=[payload.mp_id], fields=fields, chunk_size=payload.limit or 3)
            elif payload.formula:
                docs = mpr.materials.summary.search(formula=payload.formula, fields=fields, chunk_size=payload.limit or 3)
            elif payload.elements:
                elements = payload.elements if isinstance(payload.elements, list) else [str(payload.elements)]
                docs = mpr.materials.summary.search(elements=elements, fields=fields, chunk_size=payload.limit or 3)
            else:
                raise HTTPException(status_code=400, detail="Provide 'formula' or 'mp_id' or 'elements'")

            for d in list(docs)[: payload.limit or 3]:
                try:
                    row = d.model_dump()
                except Exception:
                    row = dict(d)
                rid = row.get("material_id")
                url = f"https://materialsproject.org/materials/{rid}" if rid else None
                # normalize spacegroup symbol
                spg = None
                sg = row.get("spacegroup")
                if isinstance(sg, dict):
                    spg = sg.get("symbol")
                results.append({
                    "material_id": rid,
                    "formula_pretty": row.get("formula_pretty"),
                    "e_above_hull": row.get("e_above_hull"),
                    "formation_energy_per_atom": row.get("formation_energy_per_atom"),
                    "band_gap": row.get("band_gap"),
                    "is_metal": row.get("is_metal"),
                    "density": row.get("density"),
                    "spacegroup_symbol": spg,
                    "url": url,
                })
        return {"success": True, "data": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MP proxy error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8093)


