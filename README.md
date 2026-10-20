# Deep Generative Image Impainting
https://arxiv.org/pdf/2201.09865


## Budget-aware RePaint extension (research hypothesis)

The fixed RePaint baseline (`src/repaint.py`, `scripts/run_local.py`) is unchanged.
An optional extension reallocates RePaint's resampling cycles under the same
denoiser-call budget (2,410 NFE for T=250, j=10, r=10) using a pilot pass:

* code: `src/adaptive_repaint.py`, `src/experiment_io.py`
* notebooks: `Notebooks/01_repaint_baseline_benchmark.ipynb`, `Notebooks/02_budget_aware_adaptive_repaint.ipynb`
* method and references: `Docs/budget_aware_repaint_methods.md`; audit and test log: `Docs/budget_aware_change_report.md`
* tests: `python -m pytest tests -q`

No quality improvement has been demonstrated yet; see the methods document.
