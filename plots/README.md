# Plots and figure assets

This folder holds **static** charts (SVG/PNG) and CSVs used for papers or slides. They are not generated automatically on every training run.

## Regenerating pipeline figures

Training scripts in `train_eval` can write PDF/PNG when you set a plot directory:

```text
cd train_eval
python qlattice_standalone.py --mode qcr_labels --plot-dir ../plots/run_2026-01-01
python tabpfn_standalone.py --mode qcr_labels --plot-dir ../plots/run_2026-01-01
```

### Prediction CSV exports (Q_meas + Q_cr or Q_gas)

By default, each finished QLattice / TabPFN run writes a CSV next to the plots:

- **Filename:** `{qlattice|tabpfn}_{mode}_seed{seed}.csv` (and `_cv_last_fold` when using K-fold + last-fold reference).
- **Columns:** `split`, `is_train`, `loading` (-1/0/1), **`Test status`** (`Unloaded` / `Near L.U` / `Loaded`, same as the well CSV), `Q_meas_m3day`, `Q_cr_pred_m3day` or `Q_gas_pred_m3day`, optional `delta_pos_ref_m3day` / `delta_neg_ref_m3day`, `source_csv`.

To save elsewhere: `--export-csv-dir path/to/folder`. To disable: `--no-export-csv`.

Reload and plot: `train_eval/performance_plots_from_csv.ipynb` (uses `src/prediction_csv_tools.py`).

Use a non-interactive matplotlib backend on servers:

```text
set MPLBACKEND=Agg
```

(QLattice / TabPFN call `plot_turner_qmeas_vs_qrate` and `plot_residual_qminus_qmeas` in `train_eval/src/qlattice_pipeline.py`.)

### Multi-method comparison (`Q_cr` on x, `Q_meas` on y)

`plot_qcr_x_qmeas_y_multimethod` overlays several models’ predicted critical rates on the **x-axis** vs measured gas rate on the **y-axis**, with optional linear trends, a **1:1** reference line, **near-diagonal** ring highlights, and light vertical bands by **test status** (p5–p95 of `Q_cr` per class, using the first method for the x-range). See `train_eval/compare_qcr_methods_plot.ipynb`.

## Notebooks

- `wellbore_plot.ipynb` — optional scratch space for wellbore sketches; main evaluation plots come from the pipelines above.
