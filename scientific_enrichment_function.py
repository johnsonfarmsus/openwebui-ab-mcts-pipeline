"""
title: Scientific Data Enrichment
author: Trevor Johnson
version: 1.0.0
description: Enriches chemistry and materials science queries with data from PubChem, ChEMBL, Materials Project, and RDKit
required_open_webui_version: 0.3.0
"""

import os
import re
import httpx
import json
from typing import Optional, Dict, Any, List, Callable, Awaitable
from pydantic import BaseModel, Field


class Filter:
    """Scientific Data Enrichment Filter for Open WebUI"""

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations."
        )
        ENRICHMENT_API_URL: str = Field(
            default="http://scientific-enrichment-api:8099",
            description="URL of the Scientific Enrichment API service"
        )
        MATERIALS_PROJECT_API_KEY: str = Field(
            default="",
            description="Materials Project API key (optional, but recommended for best results)"
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=False,
            description="Enable scientific data enrichment for this chat"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Inlet filter - enriches chemistry/materials queries before sending to LLM.
        This is called automatically for every query when the filter is enabled.
        """
        # Check if user has enabled enrichment for this chat
        if __user__ and __user__.get("valves"):
            if not __user__["valves"].enabled:
                return body
        else:
            # If no user valves set, don't enrich (default off)
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Get the last user message
        last_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_message = msg
                break

        if not last_message:
            return body

        query = last_message.get("content", "")
        if not query:
            return body

        # Enrich the query
        enriched = await self._enrich_query(query)

        # Update the last user message with enriched content
        last_message["content"] = enriched

        return body

    async def _enrich_query(self, query: str) -> str:
        """
        Enriches chemistry and materials science queries with contextual data.

        Automatically detects:
        - Chemical formulas (e.g., H2O, H2SO4, LiFePO4)
        - Drug/chemical names (e.g., ibuprofen, aspirin)
        - Materials Project IDs (e.g., mp-149)

        Returns enriched context including:
        - Materials properties (band gap, density, formation energy)
        - Chemical structures and properties (SMILES, molecular weight)
        - Drug information (bioactivity, targets, indications)
        - Molecular descriptors (Lipinski rules, logP, TPSA)

        :param query: The user's query to enrich
        :return: Enriched context string to prepend to the query
        """
        enrichment_parts = []

        # Extract Materials Project ID
        mp_match = re.search(r'\b(mp-\d+)\b', query, re.IGNORECASE)

        # Extract chemical formulas
        formulas = self._extract_chemical_formulas(query)

        # Extract potential chemical/drug names
        chemical_names = self._extract_chemical_names(query)

        async with httpx.AsyncClient(timeout=20.0) as client:
            # Materials Project lookups
            if mp_match:
                mp_id = mp_match.group(1)
                await self._lookup_materials_project(client, {"mp_id": mp_id}, enrichment_parts)
            elif formulas:
                for formula in formulas[:3]:  # Limit to 3 formulas
                    await self._lookup_materials_project(client, {"formula": formula}, enrichment_parts)

            # Chemical name lookups (PubChem + ChEMBL + RDKit)
            for name in chemical_names[:3]:  # Limit to 3 names
                await self._lookup_chemical(client, name, enrichment_parts)

        if enrichment_parts:
            context = "**Scientific Context:**\n" + "\n".join(f"- {part}" for part in enrichment_parts)
            return f"{context}\n\n**User Query:** {query}"

        return query

    def _extract_chemical_formulas(self, text: str) -> List[str]:
        """Extract chemical formulas from text (e.g., H2O, LiFePO4)"""
        formula_pattern = r"\b(?:[A-Z][a-z]?\d{0,3}){1,6}\b"
        matches = re.findall(formula_pattern, text)

        # Filter out common English words
        common_words = {"I", "A", "In", "As", "At", "Be", "He", "No", "If", "So", "To", "On", "It", "Am", "An"}
        formulas = []
        seen = set()

        for match in matches:
            if match in common_words or len(match) < 2:
                continue
            if match in seen:
                continue
            if not any(c.isupper() for c in match):
                continue
            seen.add(match)
            formulas.append(match)

        return formulas

    def _extract_chemical_names(self, text: str) -> List[str]:
        """Extract potential chemical/drug names from text"""
        words = re.findall(r'\b[a-z]{4,15}\b', text.lower())

        common_words = {
            "what", "when", "where", "which", "about", "their", "there", "would",
            "could", "should", "these", "those", "with", "from", "have", "that",
            "this", "they", "will", "your", "been", "were", "said", "each",
            "also", "more", "some", "into", "just", "know", "make", "than",
            "such", "only", "very", "even", "back", "good", "much", "well",
            "does", "most", "many", "happens", "happen", "combine", "combining",
            "mixed", "mixing", "together", "result", "results", "hazardous",
            "dangerous", "safe", "react", "reaction", "chemical", "chemicals",
        }

        candidates = []
        for word in words:
            if word not in common_words and len(word) >= 5:
                candidates.append(word)

        return candidates[:3]  # Limit to 3

    async def _lookup_materials_project(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, str],
        enrichment_parts: List[str]
    ):
        """Lookup materials from Materials Project"""
        try:
            r = await client.post(f"{self.valves.ENRICHMENT_API_URL}/tools/materials/lookup", json=payload)
            if r.status_code == 200:
                result = r.json()
                if result.get("success") and result.get("data"):
                    data = result["data"]
                    if isinstance(data, list) and len(data) > 0:
                        mat = data[0]
                    else:
                        mat = data

                    # Extract relevant properties
                    formula = mat.get("formula_pretty") or payload.get("formula") or payload.get("mp_id")
                    parts = [f"**Materials Project ({formula})**"]

                    if mat.get("material_id"):
                        parts.append(f"ID: {mat['material_id']}")
                    if mat.get("band_gap") is not None:
                        parts.append(f"Band gap: {mat['band_gap']:.2f} eV")
                    if mat.get("density") is not None:
                        parts.append(f"Density: {mat['density']:.2f} g/cm³")
                    if mat.get("formation_energy_per_atom") is not None:
                        parts.append(f"Formation energy: {mat['formation_energy_per_atom']:.3f} eV/atom")
                    if mat.get("e_above_hull") is not None:
                        parts.append(f"Energy above hull: {mat['e_above_hull']:.3f} eV/atom")
                    if mat.get("is_metal") is not None:
                        parts.append(f"Metal: {'Yes' if mat['is_metal'] else 'No'}")

                    spacegroup = mat.get("spacegroup_symbol") or (mat.get("spacegroup") or {}).get("symbol")
                    if spacegroup:
                        parts.append(f"Space group: {spacegroup}")

                    enrichment_parts.append(" | ".join(parts))
        except Exception as e:
            pass  # Silently fail for enrichment

    async def _lookup_chemical(
        self,
        client: httpx.AsyncClient,
        name: str,
        enrichment_parts: List[str]
    ):
        """Lookup chemical from PubChem, ChEMBL, and calculate RDKit descriptors"""
        try:
            # PubChem lookup
            r = await client.post(f"{self.valves.ENRICHMENT_API_URL}/tools/pubchem/lookup", json={"name": name})
            if r.status_code != 200:
                return

            pubchem_data = r.json()
            if not pubchem_data.get("success"):
                return

            parts = [f"**PubChem ({name})**"]

            # Add IUPAC name if available
            if pubchem_data.get("iupac"):
                parts.append(f"IUPAC: {pubchem_data['iupac'][:50]}...")

            # Add molecular formula and weight
            props = pubchem_data.get("properties", {})
            if props.get("MolecularFormula"):
                parts.append(f"Formula: {props['MolecularFormula']}")
            if props.get("MolecularWeight"):
                parts.append(f"MW: {props['MolecularWeight']}")

            enrichment_parts.append(" | ".join(parts))

            # RDKit calculations if SMILES available
            smiles = pubchem_data.get("smiles")
            if smiles:
                rdkit_r = await client.post(
                    f"{self.valves.ENRICHMENT_API_URL}/tools/chem/lipinski_pains",
                    json={"smiles": smiles}
                )
                if rdkit_r.status_code == 200:
                    rdkit_data = rdkit_r.json()
                    if rdkit_data.get("success") and rdkit_data.get("lipinski"):
                        lip = rdkit_data["lipinski"]
                        rdkit_parts = [f"**RDKit ({name})**"]
                        rdkit_parts.append(f"MW: {lip.get('mw', 0):.1f}")
                        rdkit_parts.append(f"HBD: {lip.get('hbd', 0)}")
                        rdkit_parts.append(f"HBA: {lip.get('hba', 0)}")
                        rdkit_parts.append(f"LogP: {lip.get('logp', 0):.1f}")
                        rdkit_parts.append(f"TPSA: {lip.get('tpsa', 0):.1f}")
                        rdkit_parts.append(f"Lipinski: {'✓' if lip.get('passes') else '✗'}")
                        enrichment_parts.append(" | ".join(rdkit_parts))

            # ChEMBL lookup for drug information
            chembl_r = await client.post(f"{self.valves.ENRICHMENT_API_URL}/tools/chembl/lookup", json={"name": name})
            if chembl_r.status_code == 200:
                chembl_data = chembl_r.json()
                if chembl_data.get("success"):
                    chembl_parts = [f"**ChEMBL ({name})**"]

                    if chembl_data.get("pref_name"):
                        chembl_parts.append(f"Drug: {chembl_data['pref_name']}")

                    # Clinical phase
                    if chembl_data.get("max_phase"):
                        phase = chembl_data["max_phase"]
                        phase_map = {4: "Approved", 3: "Phase 3", 2: "Phase 2", 1: "Phase 1", 0: "Preclinical"}
                        phase_desc = phase_map.get(int(float(phase)), f"Phase {phase}")
                        chembl_parts.append(f"Status: {phase_desc}")

                    # Indications
                    if chembl_data.get("indications"):
                        indications = [
                            ind.get("mesh_heading") or ind.get("efo_term")
                            for ind in chembl_data["indications"][:2]
                        ]
                        indications = [i for i in indications if i]
                        if indications:
                            chembl_parts.append(f"Treats: {', '.join(indications)}")

                    # Targets
                    if chembl_data.get("bioactivity_targets"):
                        targets = [t.get("target_id") for t in chembl_data["bioactivity_targets"][:2]]
                        if targets:
                            chembl_parts.append(f"Targets: {', '.join(targets)}")

                    if len(chembl_parts) > 1:  # Only add if we have more than just the header
                        enrichment_parts.append(" | ".join(chembl_parts))

        except Exception as e:
            pass  # Silently fail for enrichment
