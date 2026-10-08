# Predictive Maintenance: RUL Prediction for Jet Engines (LSTM/GRU)

All numbers below were produced by `python -m src.run_all` (seed 42); see `reports/results/`.

## 1. Problem
Unplanned engine failures are costly and unsafe; fixed-schedule maintenance wastes life of healthy engines.
We predict the Remaining Useful Life (RUL, in cycles) of turbofan engines from multivariate sensor time series.

## 2. Objective
Alert before failure, avoid alerting too early, and quantify the benefit as a cost comparison of run-to-failure,
fixed-schedule and model-driven maintenance.

## 3. Dataset
NASA C-MAPSS FD001: 100 training engines run to failure, 100 test engines cut off early with true RUL in `RUL_FD001.txt`.
26 columns: engine id, cycle, 3 operational settings, 21 sensors. Training RUL = max_cycle - cycle, clipped at 125.

## 4. Preprocessing
Dropped constant sensors (s1, s5, s6, s10, s16, s18, s19) and operational settings -> 14 features.
MinMaxScaler fit on training engines only. Sliding windows of 30 cycles; short test sequences are front-padded;
test engines use only their last window. Train/validation split 80/20 **by engine id**.
EDA figures: `reports/figures/sensor_trends.png`, `correlation_heatmap.png`, `sensor_std.png`.

## 5. Model selection
Baselines: Linear Regression and Random Forest on flattened windows.
LSTM: Input(30, 14) -> LSTM(64, seq) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1);
GRU with the same shape. Gated units retain gradual degradation trends and ignore noise, unlike a single-snapshot model.

## 6. Development
MSE loss, Adam (lr 1e-3), batch 128, up to 60 epochs, EarlyStopping (patience 8, restore best), ReduceLROnPlateau (x0.5, patience 4).
Loss curves: `reports/figures/loss_curves.png`.

## 7. Evaluation (FD001 test set, 100 engines)
| model            |   RMSE |    MAE |    R2 |   NASA_score |   epochs_run |   train_sec |
|:-----------------|-------:|-------:|------:|-------------:|-------------:|------------:|
| LinearRegression | 16.204 | 13.068 | 0.848 |      421.036 |          nan |         0.9 |
| RandomForest     | 18.209 | 13.571 | 0.808 |      809.994 |          nan |        14.2 |
| LSTM             | 15.263 | 11.785 | 0.865 |      339.529 |           34 |       533.1 |
| GRU              | 14.705 | 11.072 | 0.875 |      351.573 |           17 |       249.3 |

The best recurrent model (GRU, RMSE 14.71) beats the best baseline (LinearRegression, RMSE 16.20).

![comparison](reports/figures/model_comparison.png)
![pred vs true](reports/figures/pred_vs_true.png)
![trajectories](reports/figures/rul_trajectories.png)
![errors](reports/figures/error_distribution.png)

### Ablations (window size, LSTM vs GRU, RUL clipping)
| model   |   window | clipped   |   RMSE |    MAE |    R2 |   NASA_score |
|:--------|---------:|:----------|-------:|-------:|------:|-------------:|
| LSTM    |       20 | True      | 17.419 | 12.001 | 0.824 |      684.481 |
| LSTM    |       30 | True      | 15.263 | 11.785 | 0.865 |      339.529 |
| LSTM    |       50 | True      | 14.708 | 10.55  | 0.875 |      338.984 |
| LSTM    |       30 | False     | 23.983 | 16.537 | 0.667 |     7630.7   |
| GRU     |       20 | True      | 17.252 | 12.042 | 0.828 |      712.008 |
| GRU     |       30 | True      | 14.705 | 11.072 | 0.875 |      351.573 |
| GRU     |       50 | True      | 14.836 | 10.723 | 0.873 |      368.276 |
| GRU     |       30 | False     | 23.083 | 16.091 | 0.691 |     5478.38  |

## 8. Business interpretation
Costs (editable in `src/config.py`): unplanned failure 100, planned maintenance 10,
wasted life 0.2 per unused cycle. Simulated on the held-out validation engines run to failure
(per-cycle predictions; alert the first time predicted RUL < T).

Threshold sweep (default T=30, recommended T=20):
|   T |   caught |   early |   late |   missed |   total_cost |
|----:|---------:|--------:|-------:|---------:|-------------:|
|  10 |       10 |       0 |     10 |        0 |       1126   |
|  15 |       19 |       0 |      1 |        0 |        342.8 |
|  20 |       20 |       0 |      0 |        0 |        273.6 |
|  25 |       20 |       0 |      0 |        0 |        288.4 |
|  30 |       20 |       0 |      0 |        0 |        306   |
|  35 |       19 |       1 |      0 |        0 |        335.6 |
|  40 |       19 |       1 |      0 |        0 |        359.6 |
|  45 |       12 |       8 |      0 |        0 |        392.6 |
|  50 |        5 |      15 |      0 |        0 |        424.2 |
|  55 |        1 |      19 |      0 |        0 |        445.2 |
|  60 |        1 |      19 |      0 |        0 |        457.6 |

At T=20: 20 caught, 0 early, 0 late, 0 missed; total cost 273.6.

| strategy                     |   total_cost |
|:-----------------------------|-------------:|
| Run-to-failure               |       2000   |
| Fixed schedule (S=155)       |        578.6 |
| Fixed schedule, oracle S=134 |        478   |
| Model-driven (T=20)          |        273.6 |

![threshold sweep](reports/figures/threshold_sweep.png)

## 9. Conclusion and limitations
The best recurrent model (GRU, RMSE 14.71) beats the best baseline (LinearRegression, RMSE 16.20). The cost simulation uses only 20 validation engines, so cost differences between
thresholds are noisy; the oracle fixed schedule is optimistic because it is tuned on the same engines.
Test RUL is evaluated against the unclipped true RUL from `RUL_FD001.txt`.

**Training stability note.** In a first run, training on raw 0-125 targets made the LSTM (seed 42, window 30) stall on the
"predict the mean" plateau (test RMSE 41.80, R2 ~ 0); a check with other seeds showed it was seed-dependent (seed 1: RMSE 15.12, seed 2: 41.80).
Training on targets scaled to [0, 1] (RUL/125, rescaled inside the saved model) fixed this, and all reported results use that setup.
This was found and fixed after seeing test results, so the choice is not independent of the test set. All results are single-seed.
The LSTM took the longest to train (see `train_sec`), close to the 10-minute CPU budget on this laptop.

