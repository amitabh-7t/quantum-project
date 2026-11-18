# quantum-project

**Quantum Kernels Underperform Classical Machine Learning for Solar PV Fault Detection: Evidence from SCADA Time-Series Data** (SANKET 2026, paper 192)

This repository is the rebuilt experiment pipeline for that paper. It compares a classical RBF-SVM (and Isolation Forest / One-Class SVM baselines) with a quantum support vector machine that uses simulated quantum feature maps on weakly labelled 15-minute solar-PV SCADA time series from two plants. Quantum kernels are simulated with PennyLane's `default.qubit` statevector device; the noise sweep uses `default.mixed`. No results here come from physical quantum hardware.

The reported numbers were produced with Python 3.13.5 and the package versions in `requirements.txt`.

## Data

Download the Kaggle dataset **Solar Power Generation Data** and copy these four files into `data/`:

- `Plant_1_Generation_Data.csv`
- `Plant_1_Weather_Sensor_Data.csv`
- `Plant_2_Generation_Data.csv`
- `Plant_2_Weather_Sensor_Data.csv`

Then build the cached, labelled table used by the experiments:

```bash
pip install -r requirements.txt
cd src
python build_labeled.py
```

This writes `data/labeled.pkl` (gitignored). The experiment scripts load that file.

## How to run

Run the scripts from `src/` so the local imports resolve. Paths to `data/`, `results/`, and `figures/` are relative to the repository root, not the working directory.

| Script | What it does |
| --- | --- |
| `python build_labeled.py` | Build `data/labeled.pkl` from the four CSVs (features, weak labels, `hour`, `day`). |
| `python data.py` | Load each plant's CSVs and print row counts, inverter counts, dates, and median AC/DC (daylight only). |
| `python feats.py` | Build labels for variants A and B and print per-plant counts of anomalies and the three rules. |
| `python qkernel.py` | Check that the statevector kernel matches the pairwise compute-uncompute circuit, and that a zero-noise mixed-state kernel matches the exact kernel. |
| `python classical_full.py` | Fit Isolation Forest, One-Class SVM, and RBF-SVM on the full 75/25 stratified split and print test metrics. |
| `python matched.py` | Seed-42 matched 150/80 subsample: CV-tuned QSVM vs RBF-SVM, plus i.i.d. and day-block bootstrap of the F1 gap. |
| `python exp_seeds.py [configs...]` | 20-seed (10 for `noise`) matched comparisons. With no arguments, runs every config below. Writes `results/res_seeds_<config>.csv`. |
| `python exp_full.py` | Full-data classical baselines, permutation importance, seed-42 matched details, common test-set comparison, chronological and cross-plant generalisation, DC-correction ablation, and threshold sensitivity. Writes `results/results_full.json`. |
| `python exp_cost.py` | Kernel-construction, training, and inference timings (pairwise circuits vs statevector shortcut vs classical grid search), plus scaling in training size `n`. Writes `results/results_cost.json`. |
| `python exp_sens_matched.py` | Threshold sensitivity of the matched classical–QSVM F1 gap (10 seeds per threshold configuration). Writes `results/res_sens_matched.csv`. |
| `python figs.py` | Rebuild the paper figures into `figures/` from `results/` (and `data/labeled.pkl` for the ROC panel). |

`exp_seeds.py` configs: `base`, `zz_L1to4`, `zonly`, `iqp_allpairs`, `pca2`, `pca6`, `pca8`, `noleak`, `raw`, `chrono`, `chrono_noleak`, `p1to2`, `p2to1`, `noise`. Example: `python exp_seeds.py base noleak chrono`.

The seed sweeps, `exp_full.py`, `exp_cost.py`, and `exp_sens_matched.py` are the long runs. Precomputed logs, CSVs, and JSON from the paper's rerun are already in `results/`.

## Results

These figures are taken from the files in `results/`, not re-computed here.

Headline, 20 random matched splits (`results/logs_seeds.txt`, `results/res_seeds_base.csv`): **SVC F1 0.906±0.048 vs QSVM F1 0.759±0.105**. Classical wins 20/20. Mean AUC is 0.976 (SVC) and 0.888 (QSVM).

Other 20-split comparisons from the same log:

| Setting | SVC F1 | QSVM F1 | Classical wins |
| --- | --- | --- | --- |
| No-leakage features | 0.747±0.064 | 0.656±0.071 | 17/20 |
| Chronological split | 0.943±0.030 | 0.705±0.131 | 20/20 |
| Plant 1 → Plant 2 | 0.880±0.076 | 0.500±0.160 | 20/20 |
| Plant 2 → Plant 1 | 0.614±0.222 | 0.543±0.130 | 15/20 |
| PCA 2 / 6 / 8 components | 0.926 / 0.944 / 0.935 | 0.793 / 0.662 / 0.501 | 18/20, 20/20, 20/20 |
| Z-only map | 0.906±0.048 | 0.910±0.058 | 7/20 |
| IQP all-pairs | 0.906±0.048 | 0.772±0.074 | 20/20 |

Per-layer test F1 on the ZZ map (`results/logs_perL.txt`, from `res_seeds_zz_L1to4.csv`): 0.780±0.068 at L=1, falling to 0.568±0.106 at L=4.

Noise (`results/logs_seeds.txt`, 10 seeds): shots-100 / shots-1000 / depolarising 0.001 / 0.01 F1 means 0.644, 0.652, 0.654, 0.660.

Cost (`results/results_cost.json`): the pairwise quantum kernel used 23,325 circuits and 406.7 s (train 198.4 s + test 208.3 s); the classical RBF-SVM grid search on the same 150-point subsample took 1.60 s.

Figures from this rerun are in `figures/` (`fig_pipeline.pdf`, `fig_layer_sweep.pdf`, `fig_confusion.pdf`, `fig_roc.pdf`, `fig_feature_importance.pdf`).
