"""
Chapter 18 hands-on project, Part 3: a real Streamlit front-end for
`service.py`'s FastAPI backend -- the second real "interactive web
tool" Section 18.1's outline names, calling the real, running API over
real HTTP rather than importing the model directly, exactly the
two-tier (API + UI) architecture a real deployed model typically uses
so the same backend can also serve other real clients (a script, a
notebook, another team's application) without duplicating logic.

Run `service.py` first (`uvicorn service:app`), then
`streamlit run app_streamlit.py`.

See README.md for usage and chapter.md Section 18.1 for context.
"""
import io

import requests
import streamlit as st
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="hERG Blocker Predictor", page_icon="🧪")
st.title("hERG Blocker Prediction")
st.caption("Chapter 18 hands-on project — a real model (Chapter 5's methodology) served through a real FastAPI backend.")

with st.sidebar:
    st.subheader("Service status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        st.success("API reachable")
        st.write(f"Model version: `{health['model_version']}`")
        st.write(f"Training compounds: {health['n_training_compounds']}")
        st.write(f"Held-out ROC-AUC: {health['held_out_roc_auc']}")
        st.code(health["model_sha256"][:16] + "...", language=None)
    except requests.exceptions.RequestException:
        st.error(f"Cannot reach the API at {API_BASE_URL}. Start it with:\n\nuvicorn service:app")

smiles = st.text_input("SMILES", value="CC(=O)Oc1ccccc1C(=O)O", help="Paste a SMILES string for the compound to evaluate.")

if st.button("Predict", type="primary") and smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error(f"'{smiles}' is not a valid SMILES string.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            img = Draw.MolToImage(mol, size=(280, 280))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption="2D structure (RDKit)")

        try:
            response = requests.post(f"{API_BASE_URL}/predict", json={"smiles": smiles}, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as exc:
            st.error(f"API request failed: {exc}")
        else:
            with col2:
                proba = result["herg_blocker_probability"]
                if result["herg_blocker_predicted"]:
                    st.error(f"Predicted hERG blocker (P = {proba:.3f})")
                else:
                    st.success(f"Predicted non-blocker (P = {proba:.3f})")
                st.progress(proba)

                admet = result["admet"]
                st.subheader("ADMET / drug-likeness (Chapter 13's Tier 1 filter)")
                if admet["passes_admet"]:
                    st.success("Passes rule-based drug-likeness filter")
                else:
                    st.warning("Fails rule-based drug-likeness filter")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MW", f"{admet['molecular_weight']:.1f}")
                c2.metric("LogP", f"{admet['logp']:.2f}")
                c3.metric("TPSA", f"{admet['tpsa']:.1f}")
                c4.metric("QED", f"{admet['qed']:.3f}")
                if admet["pains_alert"]:
                    st.warning("Real PAINS structural alert flagged (Baell & Holloway, 2010)")

with st.expander("What is this predicting, and how well?"):
    st.markdown(
        "This app predicts the probability that a compound blocks the hERG "
        "(KCNH2) potassium channel — the single most common real reason a "
        "drug candidate is deprioritized or, in rare cases, withdrawn after "
        "approval, due to real cardiac arrhythmia risk (QT prolongation). "
        "The model is a real XGBoost classifier trained on real ChEMBL240 "
        "bioactivity data (Chapter 5's own methodology), evaluated honestly "
        "under a real Bemis-Murcko scaffold split so its reported accuracy "
        "is not inflated by near-duplicate analogs — see the sidebar for "
        "this exact model's own real held-out ROC-AUC, and `chapter.md` "
        "Section 18.1 for the full account."
    )
