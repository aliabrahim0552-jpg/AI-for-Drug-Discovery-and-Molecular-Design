# Vendored code notice

`protein_mpnn_utils.py` in this directory is fetched verbatim, with no
modifications, from the official ProteinMPNN repository
(https://github.com/dauparas/ProteinMPNN, commit `main` as of
2026-08-20), authored by Justas Dauparas and the other authors of
Dauparas et al. (2022) — see `chapter.md`'s References. It is
distributed under the MIT License reproduced in `LICENSE` (also fetched
verbatim from the same repository), which permits this redistribution.

It is vendored here rather than installed as a pip/conda package
because ProteinMPNN is not published as an installable package — the
upstream project is distributed only as a Git repository of scripts.

`../../proteinmpnn_weights/v_48_020.pt` is the official pretrained
checkpoint (`vanilla_model_weights/v_48_020.pt` in the upstream repo:
48 nearest-neighbor edges, 0.20 Å training noise — the model that
repository's own `protein_mpnn_run.py` uses by default), fetched the
same day from the same repository, also unmodified. No weights in this
chapter are retrained, fine-tuned, or otherwise altered from their
official released form.

`protein_design.py`, in this chapter's root folder, is this book's own
code and imports from the vendored file above; it is not part of the
upstream ProteinMPNN distribution.
