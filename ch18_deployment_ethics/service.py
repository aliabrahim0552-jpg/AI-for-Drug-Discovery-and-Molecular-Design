"""
Chapter 18 hands-on project, Part 2: a real FastAPI web service
serving the real, versioned hERG-blocker model `train_model.py` built
-- "packaging models into interactive web tools" (Section 18.1),
concretely rather than as a theoretical description. Loads the real
saved artifact once at startup (with its own real SHA-256 provenance
check), then answers real, structured predictions over HTTP: for a
submitted real SMILES string, the real predicted hERG-blocker
probability alongside the real, established Tier 1 rule-based
drug-likeness filter Chapter 13 introduced (Lipinski Ro5, Veber's
rules, PAINS, QED) -- so one request returns the same two real,
complementary signals (a learned property prediction and a rule-based
filter) this book has used together since Chapter 16's own capstone
pipeline, now exposed as a real, callable API rather than a one-off
script.

See README.md for usage and chapter.md Section 18.1 for context.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from train_model import featurize, load_deployed_model

RDLogger.DisableLog("rdApp.*")

_MODEL = None
_METADATA = None
_PAINS_CATALOG = None

# Rule-based ADMET thresholds -- identical to Chapter 13's own Tier 1
# filter (Lipinski Ro5, Veber's rules, PAINS, QED), reused unchanged.
LIPINSKI_MAX_MW = 500.0
LIPINSKI_MAX_LOGP = 5.0
LIPINSKI_MAX_HBD = 5
LIPINSKI_MAX_HBA = 10
LIPINSKI_MAX_VIOLATIONS = 1
VEBER_MAX_ROTB = 10
VEBER_MAX_TPSA = 140.0
QED_MIN = 0.30


def _pains_catalog() -> FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog(params)
    return _PAINS_CATALOG


def compute_admet_flags(mol: Chem.Mol) -> dict:
    mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
    tpsa, rotb = Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol)
    qed = QED.qed(mol)
    pains_alert = bool(_pains_catalog().HasMatch(mol))
    lipinski_violations = sum([mw > LIPINSKI_MAX_MW, logp > LIPINSKI_MAX_LOGP, hbd > LIPINSKI_MAX_HBD, hba > LIPINSKI_MAX_HBA])
    veber_pass = rotb <= VEBER_MAX_ROTB and tpsa <= VEBER_MAX_TPSA
    return {
        "molecular_weight": round(mw, 2), "logp": round(logp, 3), "tpsa": round(tpsa, 2),
        "hbd": hbd, "hba": hba, "rotatable_bonds": rotb, "qed": round(qed, 4),
        "pains_alert": pains_alert, "lipinski_violations": lipinski_violations,
        "passes_admet": bool(lipinski_violations <= LIPINSKI_MAX_VIOLATIONS and veber_pass and not pains_alert and qed >= QED_MIN),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _MODEL, _METADATA
    _MODEL, _METADATA = load_deployed_model()
    yield


app = FastAPI(
    title="hERG Blocker Prediction Service",
    description="Chapter 18 hands-on project: a real FastAPI service serving a real, versioned XGBoost hERG-liability classifier (Chapter 5's methodology) plus Chapter 13's real ADMET filter.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    smiles: str = Field(..., description="A SMILES string for the query compound.", examples=["CC(=O)Oc1ccccc1C(=O)O"])


class PredictResponse(BaseModel):
    smiles: str
    canonical_smiles: str
    herg_blocker_probability: float
    herg_blocker_predicted: bool
    admet: dict
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_sha256: str
    n_training_compounds: int
    held_out_roc_auc: float


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=_METADATA["model_version"],
        model_sha256=_METADATA["sha256_model_file"],
        n_training_compounds=_METADATA["n_training_compounds"],
        held_out_roc_auc=_METADATA["held_out_scaffold_split_metrics"]["roc_auc"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    mol = Chem.MolFromSmiles(request.smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"'{request.smiles}' could not be parsed as a valid SMILES string.")
    canonical_smiles = Chem.MolToSmiles(mol)

    features = featurize([canonical_smiles])
    proba = float(_MODEL.predict_proba(features)[0, 1])
    admet = compute_admet_flags(mol)

    return PredictResponse(
        smiles=request.smiles,
        canonical_smiles=canonical_smiles,
        herg_blocker_probability=round(proba, 4),
        herg_blocker_predicted=bool(proba >= 0.5),
        admet=admet,
        model_version=_METADATA["model_version"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("service:app", host="127.0.0.1", port=8000, reload=False)
