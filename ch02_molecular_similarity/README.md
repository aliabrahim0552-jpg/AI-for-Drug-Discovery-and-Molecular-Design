# Chapter 2 Hands-on: Molecular Similarity Search

Ranks a small compound library by Tanimoto similarity (Morgan/ECFP4
fingerprints) to a query molecule, and flags which candidates pass
Lipinski's Rule of Five. See [`chapter.md`](chapter.md) Section 2.5 for
full context, and the chapter as a whole for the representation,
fingerprint, and similarity-metric background this project builds on.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Default query is Aspirin
python similarity_search.py

# Query with your own molecule (SMILES) and show top 5
python similarity_search.py --query "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --top 5
```

`molecules.csv` holds the demo library (17 well-known drugs and
biomolecules). Swap in a real ChEMBL/PubChem extract to scale this up —
that's the Chapter 1 & 4 project in the book outline.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Google Colab

`rdkit` installs with a single `pip install rdkit` on Colab's default
runtime; everything else is standard library. No GPU required.
