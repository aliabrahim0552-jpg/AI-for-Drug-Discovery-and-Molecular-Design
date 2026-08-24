# AI for Drug Discovery & Molecular Design

### A Practical Guide from Sequence to Small Molecules & Biologics using Python and PyTorch

---

> **Note:** This material was developed with AI assistance and is intended for educational and learning purposes. It should not be used for direct clinical, medical, or applied laboratory applications without independent expert verification.

## About

This repository is the companion codebase for AI for Drug Discovery & Molecular Design. It covers computational representations, machine learning, generative modeling, protein structure prediction, molecular docking, and simulation workflows across 18 chapters, from foundational cheminformatics through two full capstone projects.

## Contents

**Part I: Foundations of Molecular Data & Computational Representations**
- Chapter 1: The Drug Discovery Pipeline & AI Revolution — ch01_drug_discovery_pipeline
- Chapter 2: Representing Chemical Space (Small Molecules) — ch02_molecular_similarity
- Chapter 3: Protein Representations & Structure Engineering — ch03_protein_structure
- Chapter 4: Core Toolkit & Biological Data Engineering — ch04_data_engineering

**Part II: Machine Learning & Deep Learning for Small Molecules**
- Chapter 5: QSAR & ADMET Property Modeling — ch05_qsar_admet
- Chapter 6: Graph Neural Networks (GNNs) in Chemistry — ch06_gnn_molecular_properties
- Chapter 7: Generative AI & De Novo Drug Design — ch07_generative_transformer_rl

**Part III: Protein Structure Prediction & Macromolecular Design**
- Chapter 8: Protein Language Models (pLMs) & Evolutionary Dynamics — ch08_protein_language_models
- Chapter 9: The 3D Structure Prediction Revolution — ch09_structure_prediction
- Chapter 10: De Novo Protein & Antibody Design — ch10_protein_design

**Part IV: Molecular Docking, Simulations & Virtual Screening**
- Chapter 11: AI-Driven Molecular Docking — ch11_molecular_docking
- Chapter 12: Molecular Dynamics & Neural Network Potentials — ch12_md_nnp
- Chapter 13: High-Throughput Virtual Screening (HTVS) Pipelines — ch13_htvs_pipeline

**Part V: Advanced Frontiers & Capstone Projects**
- Chapter 14: AI for RNA Therapeutics & Nucleic Acid Design — ch14_rna_therapeutics
- Chapter 15: Physics-Informed ML & Quantum Machine Learning (QML) — ch15_pinn_qml
- Chapter 16: Capstone Project 1 — Small Molecule Oncology Target Discovery — ch16_capstone_oncology
- Chapter 17: Capstone Project 2 — De Novo Neutralizing Antibody Design — ch17_antibody_design
- Chapter 18: Model Deployment, Reproducibility, Ethics & Regulatory Guidelines — ch18_deployment_ethics

See outline.md for the full chapter-by-chapter breakdown.

## Tech Stack

- Languages & Environments: Python 3.10+, Google Colab (GPU/TPU)
- Bio/Cheminformatics: RDKit, DeepChem, BioPython, OpenMM, Py3Dmol
- Machine Learning & Deep Learning: PyTorch, PyTorch Geometric, Hugging Face Transformers, Scikit-learn
- Modeling & Docking Engines: AlphaFold2/3, ESMFold, DiffDock, AutoDock Vina, RFdiffusion, ProteinMPNN

## Running the Notebooks

Each chapter folder contains self-contained, executable code designed to run on free-tier Google Colab. Open the notebook or script for a chapter and run cells top to bottom; dependencies are declared inline.

## Exports

Rendered chapter exports (DOCX/PDF) are available under exports/chapter_NN/.

## License

This project is licensed under the MIT License — see LICENSE for details.

## Contributing

Contributions are welcome — see CONTRIBUTING.md for guidelines.


## Testing

Continuous integration runs a lightweight syntax and lint check (Python compile check plus ruff) on every push and pull request. Running the full pytest suite requires a GPU-enabled environment with the specialized bio/cheminformatics and modeling dependencies installed (RDKit, DeepChem, PyTorch Geometric, OpenMM, AlphaFold/ESMFold tooling, and similar) — see each chapter's own dependencies before running its tests locally or on Google Colab.
