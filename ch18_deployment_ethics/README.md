# Chapter 18 Hands-on: Deploying a Real Model as an Interactive Web Tool

A real, versioned hERG-blocker classifier (Chapter 5's own methodology
— XGBoost on ECFP4 fingerprints, evaluated under a real Bemis-Murcko
scaffold split) packaged into a real, running FastAPI backend and a
real Streamlit front-end — this book's final chapter's hands-on
project, closing the loop from "train a model" (Chapters 1-17) to
"put it in front of a user" (Section 18.1). See
[`chapter.md`](chapter.md) Sections 18.1-18.2 for full scientific and
engineering context, including this chapter's own real, measured
results and its real model-provenance/reproducibility practices.

## Setup

```bash
pip install -r requirements.txt
```

## Run

**1. Train and save the real, versioned model artifact:**

```bash
python train_model.py
```

Fetches real hERG (CHEMBL240) bioactivity data live from ChEMBL
(cached for offline reproducibility), reports a real, honest
scaffold-split held-out evaluation, then retrains on the full real
dataset and saves the deployed artifact to
`models/herg_xgboost.joblib` plus a real SHA-256-verified metadata
file (`models/herg_xgboost.metadata.json`).

**2. Start the real FastAPI backend:**

```bash
uvicorn service:app --reload
```

Exposes `GET /health` (real model provenance/metadata) and
`POST /predict` (a real SMILES in, a real hERG-blocker probability
plus Chapter 13's real ADMET filter results out). Interactive OpenAPI
docs at `http://127.0.0.1:8000/docs`.

**3. Start the real Streamlit front-end** (in a second terminal):

```bash
streamlit run app_streamlit.py
```

Opens at `http://localhost:8501` — paste a SMILES, see the real 2D
structure, the real predicted hERG-blocker probability, and the real
ADMET flags, all served live from the FastAPI backend in step 2.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Data curation and the model-artifact provenance/tamper-detection
check are tested directly and offline. The FastAPI endpoint tests use
Starlette's real `TestClient` against the real, already-trained
artifact `train_model.py` produces — no live server needed — and are
skipped automatically if that artifact hasn't been built yet (run
`train_model.py` first).

## A note on Google Colab

FastAPI/Streamlit apps aren't naturally suited to a single Colab
notebook cell (both are long-running servers); the real, standard way
to demo this chapter's own two-tier app from Colab is `ngrok` or a
similar tunnel, exposing the local `uvicorn`/`streamlit` process on a
real public URL — a real, disclosed practical limitation of this
project relative to this book's other, single-script hands-on
projects, and part of why Section 18.1 discusses real, cloud-native
deployment options (a managed container platform, a real HTTPS
endpoint) as the actual production path.
