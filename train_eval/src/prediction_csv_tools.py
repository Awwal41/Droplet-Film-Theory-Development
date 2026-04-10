"""Reload TabPFN / QLattice prediction CSVs and reproduce Turner-style classification metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .qlattice_pipeline import predict_status, tune_delta

LOADING_NAMES = {-1: "Unloaded", 0: "Near L.U", 1: "Loaded"}


def pred_column_from_df(df: pd.DataFrame) -> str:
    """Pick Q_cr or Q_gas prediction column."""
    for c in ("Q_cr_pred_m3day", "Q_gas_pred_m3day"):
        if c in df.columns:
            return c
    raise ValueError(f"No prediction column found; columns={list(df.columns)}")


def classify_from_export(
    df: pd.DataFrame,
    *,
    pred_col: str | None = None,
    use_csv_delta: bool = True,
) -> tuple[np.ndarray, float, float]:
    """
    Apply the same Turner rule as the pipelines: tune δ on **train** rows, then classify all rows.

    If ``delta_*_ref_m3day`` are in the CSV and ``use_csv_delta`` is True, use those instead of
    re-tuning (recommended for matching the original run).
    """
    pred_col = pred_col or pred_column_from_df(df)
    y_pred = df[pred_col].to_numpy(dtype=float)
    q_meas = df["Q_meas_m3day"].to_numpy(dtype=float)
    if use_csv_delta and "delta_pos_ref_m3day" in df.columns and "delta_neg_ref_m3day" in df.columns:
        dpos = float(df["delta_pos_ref_m3day"].iloc[0])
        dneg = float(df["delta_neg_ref_m3day"].iloc[0])
    else:
        train = df[df["split"] == "train"]
        dpos, dneg = tune_delta(
            train[pred_col].to_numpy(),
            train["Q_meas_m3day"].to_numpy(),
            train["loading"].to_numpy(),
        )
    labels = predict_status(y_pred, q_meas, dpos, dneg)
    return labels, dpos, dneg


def per_class_accuracy(
    y_true: np.ndarray,
    y_hat: np.ndarray,
) -> dict[str, float]:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_hat).ravel()
    out: dict[str, float] = {}
    for k, name in LOADING_NAMES.items():
        m = yt == k
        if not np.any(m):
            out[name] = float("nan")
        else:
            out[name] = float(np.mean(yp[m] == yt[m]))
    return out


def summarize_export_csv(path: str | Path) -> pd.Series:
    """One row summary: path, overall test accuracy, per-class test accuracies."""
    path = Path(path)
    df = pd.read_csv(path)
    pred_col = pred_column_from_df(df)
    y_hat, dpos, dneg = classify_from_export(df, pred_col=pred_col)
    yt = df["loading"].to_numpy()
    split = df["split"].to_numpy()
    te = split == "test"
    overall = float(np.mean(y_hat[te] == yt[te])) if np.any(te) else float("nan")
    pc = per_class_accuracy(yt[te], y_hat[te])
    row = {
        "file": path.name,
        "pipeline": _parse_pipeline(path.name),
        "test_acc": overall,
        "delta_pos": dpos,
        "delta_neg": dneg,
    }
    row["acc_unloaded"] = pc["Unloaded"]
    row["acc_near"] = pc["Near L.U"]
    row["acc_loaded"] = pc["Loaded"]
    return pd.Series(row)


def _parse_pipeline(name: str) -> str:
    if name.startswith("qlattice"):
        return "QLattice"
    if name.startswith("tabpfn"):
        return "TabPFN"
    return "other"


def summarize_exports_dir(export_dir: str | Path, *, pattern: str = "*.csv") -> pd.DataFrame:
    """Build a summary table for every matching CSV in ``export_dir`` (e.g. after a training run)."""
    export_dir = Path(export_dir)
    paths = sorted(export_dir.glob(pattern))
    if not paths:
        return pd.DataFrame()
    return pd.DataFrame([summarize_export_csv(p) for p in paths])
