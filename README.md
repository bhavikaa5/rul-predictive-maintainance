# RUL Prediction for Jet Engines (LSTM / GRU) - NASA C-MAPSS FD001

Predicts Remaining Useful Life from sensor windows, compares against baselines, and turns predictions into a
maintenance-cost analysis plus a Streamlit demo. Runs on CPU (Colab or laptop).

## Run on Google Colab (recommended for showing the work)
1. Download FD001 (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) from the NASA Prognostics Data Repository
   (Turbofan Engine Degradation Simulation) or the Kaggle mirror "NASA CMAPSS Jet Engine Simulated Data".
2. Put them in `data/raw/`, then upload this whole folder to Google Drive as `MyDrive/rul-predictive-maintenance/`.
3. Open the notebooks from Drive in Colab and run in order: `01_eda` -> `02_modeling` -> `03_business`
   (each starts with a setup cell that mounts Drive and installs requirements). Outputs (figures, models, CSVs)
   are saved back into the Drive folder.
4. `python tools/make_report.py` (or a `!python tools/make_report.py` cell) fills `REPORT.md` with the real numbers.

## Run locally
```bash
python -m venv .venv && .venv\Scripts\activate    # source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
python -m src.run_all            # trains everything, ~ several minutes on CPU
python tools/make_report.py
streamlit run app/streamlit_app.py
```
Quick dry run: `python -m src.run_all --epochs 3 --skip-ablation`.

## Layout
`src/` data, features, models, train, evaluate, business, run_all - `notebooks/` 01_eda, 02_modeling, 03_business -
`app/streamlit_app.py` - `models/` best_model.keras, scaler.pkl, meta.json - `reports/figures`, `reports/results`.
Regenerate notebooks with `python tools/make_notebooks.py`.

## Design choices
- Targets are scaled to RUL/125 during training (rescaled inside the saved model); without this the LSTM stalled on the mean-prediction plateau.
- Seed 42 everywhere; scaler fit on training engines only; train/val split by engine id (80/20).
- Trained on clipped RUL (125); test metrics use the unclipped true RUL from `RUL_FD001.txt`.
- Business simulation runs on the held-out validation engines (full run-to-failure trajectories, per-cycle predictions),
  because test engines are truncated and have no ground truth beyond the cut-off. Outcomes: *missed* (never alerted),
  *late* (< 10 cycles left), *early* (> 50 cycles left), *caught* otherwise. Fixed schedule S = 10th percentile of training lives.

## Results
Final numbers are in `REPORT.md` and the CSVs: see `reports/results/model_comparison.csv`, `ablations.csv`, `cost_comparison.csv`, and `REPORT.md`.
(No numbers are hard-coded in this repo.)

## Limitations
FD001 only (one operating condition, one fault mode); 20 validation engines for the cost simulation (noisy);
no uncertainty estimates; single seed.
