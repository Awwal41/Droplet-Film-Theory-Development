#!/usr/bin/env python3
"""
Standalone QLattice / Feyn symbolic regression for well data (Turner-style loading).

**Reality check:** field **Q_critical** is **not observed**. This script builds **operational**
targets from **measured gas**, **labels**, and **margins** — or, optionally, from **teacher arrays
you supply** in a ``.npz``.

- **qcr_labels (usual):** No external file. Feyn fits symbolic Q_critical to **label-implied proxy
  targets** (Q_meas ± LOSS_DELTA by class; same margins as ``load_and_split``). Targets are
  constructed entirely inside this pipeline.

- **qcr_custom (optional):** Feyn fits to **your** per-row Q_cr estimates passed via
  ``--qcr-npz`` (keys ``y_train_pred`` / ``y_test_pred``, same train/test split and row order as
  this script). Use this only when you intentionally want a custom teacher; it is **not** required
  for comparison runs — use **qcr_labels** for a self-contained workflow.

- **gas:** Exploratory — predict **measured** rate only. Default: log1p(Gasflowrate);
  ``--linear-gas`` for raw rate.

Same CSV filtering + train/test split as NewTestOpti.

Install:  pip install feyn pandas numpy scikit-learn matplotlib
License: feyn community edition — see https://pypi.org/project/feyn/

Example:
  python train_eval/src/qlattice_pipeline.py
  python qlattice_standalone.py --mode qcr_labels
  python qlattice_standalone.py --mode qcr_labels --cv-folds 5
  python qlattice_standalone.py --plot-dir . --show-plots
  python qlattice_standalone.py --select-by turner_lu --loss absolute_error --epochs 36
  python qlattice_standalone.py --mode gas --cv-folds 5
  python qlattice_standalone.py --linear-gas
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _repo_root() -> Path:
    """``train_eval/src/<this file>`` → repository root (parent of ``train_eval``)."""
    return Path(__file__).resolve().parent.parent.parent


def default_processed_well_csv() -> Path:
    """Default ``datasets/processed_well_data.csv`` under the repo root."""
    return _repo_root() / "datasets" / "processed_well_data.csv"


def default_qcr_npz_path() -> Path:
    """Default path when ``--qcr-npz`` is omitted for ``qcr_custom`` / ``both``."""
    return Path(__file__).resolve().parent.parent / "custom_qcr_predictions.npz"


# Columns used after filtering (same order as NewTestOpti-style notebooks)
TURNER_FEATURE_COLS = [
    "Dia",
    "Dev(deg)",
    "Area (m2)",
    "z",
    "GasDens",
    "LiquidDens",
    "g (m/s2)",
    "P/T",
    "friction_factor",
    "critical_film_thickness",
]


# ---------------------------------------------------------------------------
# Turner δ helpers (standalone; aligned with NewTestOpti-style notebooks)
# ---------------------------------------------------------------------------
LABEL_ORDER = [1, 0, -1]

LOSS_FRAC_LOADED = 0.12
LOSS_FRAC_UNLOADED = 0.12
LOSS_FRAC_NEAR = 0.05
LOSS_FLOOR_LU = 8000.0
LOSS_FLOOR_NEAR = 3500.0

LOSS_DELTA_LOADED = 8000.0
LOSS_DELTA_UNLOADED = 8000.0
LOSS_DELTA_NEAR = 3500.0
LOSS_DELTA = 8000.0

TUNE_K_POS = 1.0
TUNE_K_NEG = 1.0
TUNE_DELTA_S_LO_FRAC = 0.04
TUNE_DELTA_S_HI_FRAC = 0.32
TUNE_OBJECTIVE = "lu_f1"

USE_DYNAMIC_DELTA = True
DYN_DELTA_POS_FRAC = 0.036
DYN_DELTA_NEG_FRAC = 0.05

def set_loss_margins(med_gas: float) -> None:
    global LOSS_DELTA_LOADED, LOSS_DELTA_UNLOADED, LOSS_DELTA_NEAR, LOSS_DELTA
    LOSS_DELTA_LOADED = max(LOSS_FLOOR_LU, float(LOSS_FRAC_LOADED) * med_gas)
    LOSS_DELTA_UNLOADED = max(LOSS_FLOOR_LU, float(LOSS_FRAC_UNLOADED) * med_gas)
    LOSS_DELTA_NEAR = max(LOSS_FLOOR_NEAR, float(LOSS_FRAC_NEAR) * med_gas)
    LOSS_DELTA = max(LOSS_DELTA_LOADED, LOSS_DELTA_UNLOADED)

def mean_f1_loaded_unloaded(y_true, y_pred):
    per = f1_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)
    return float(0.5 * (per[0] + per[2]))

def effective_delta_arrays(d_pos_ref, d_neg_ref, Q):
    Q = np.asarray(Q, dtype=float)
    if not USE_DYNAMIC_DELTA:
        return (
            np.full_like(Q, float(d_pos_ref), dtype=float),
            np.full_like(Q, float(d_neg_ref), dtype=float),
        )
    return (
        np.maximum(float(d_pos_ref), DYN_DELTA_POS_FRAC * Q),
        np.maximum(float(d_neg_ref), DYN_DELTA_NEG_FRAC * Q),
    )

def predict_status(y_pred, gsflow, d_pos=None, d_neg=None, use_dynamic=None):
    if d_pos is None:
        d_pos = LOSS_DELTA_LOADED
    if d_neg is None:
        d_neg = LOSS_DELTA_UNLOADED
    if use_dynamic is None:
        use_dynamic = USE_DYNAMIC_DELTA
    diff = np.asarray(y_pred, dtype=float) - np.asarray(gsflow, dtype=float)
    Q = np.asarray(gsflow, dtype=float)
    if not use_dynamic:
        return np.where(
            diff > d_pos,
            1,
            np.where(diff < -d_neg, -1, 0),
        )
    dp, dn = effective_delta_arrays(d_pos, d_neg, Q)
    return np.where(diff > dp, 1, np.where(diff < -dn, -1, 0))

def tune_delta(y_pred, gsflow, load_true, lo=None, hi=None, n=64, k_pos=None, k_neg=None):
    mg = float(np.nanmedian(np.asarray(gsflow, dtype=float)))
    if lo is None:
        lo = max(8000.0, TUNE_DELTA_S_LO_FRAC * mg)
    if hi is None:
        hi = max(lo * 1.25, TUNE_DELTA_S_HI_FRAC * mg)
    if k_pos is None:
        k_pos = TUNE_K_POS
    if k_neg is None:
        k_neg = TUNE_K_NEG
    true_classes = np.unique(load_true)
    best_pos = float(LOSS_DELTA_LOADED)
    best_neg = float(LOSS_DELTA_UNLOADED)
    best_score = -1.0
    candidates = []

    for s in np.linspace(lo, hi, n):
        d_pos = float(s * k_pos)
        d_neg = float(s * k_neg)
        pred = predict_status(y_pred, gsflow, d_pos, d_neg)
        f1m = f1_score(load_true, pred, average="macro", zero_division=0)
        f1_lu = mean_f1_loaded_unloaded(load_true, pred)
        bal = balanced_accuracy_score(load_true, pred)
        n_pred = len(np.unique(pred))
        if TUNE_OBJECTIVE == "lu_f1":
            score = f1_lu + 1e-4 * bal
        else:
            score = f1m + 1e-4 * bal
        candidates.append((d_pos, d_neg, score, f1m, f1_lu, n_pred, pred))

        if score > best_score:
            best_score = score
            best_pos, best_neg = d_pos, d_neg

    if len(true_classes) >= 2:
        multi = [c for c in candidates if c[5] >= 2]
        if multi:
            best = max(multi, key=lambda t: t[2])
            best_pos, best_neg = best[0], best[1]

    return best_pos, best_neg

def eval_turner_train_test_metrics(
    y_train_pred: np.ndarray,
    gs_train: np.ndarray,
    load_train: np.ndarray,
    y_test_pred: np.ndarray,
    gs_test: np.ndarray,
    load_test: np.ndarray,
) -> dict:
    """
    Tune δ on **training** predictions only, then apply the same δ to classify train (resubstitution)
    and test. Train metrics are optimistic; test metrics match the operational train→test pipeline.
    """
    dpos, dneg = tune_delta(y_train_pred, gs_train, load_train)
    pred_tr = predict_status(y_train_pred, gs_train, dpos, dneg)
    pred_te = predict_status(y_test_pred, gs_test, dpos, dneg)
    return {
        "dpos": dpos,
        "dneg": dneg,
        "acc_train": float(accuracy_score(load_train, pred_tr)),
        "f1_macro_train": float(f1_score(load_train, pred_tr, average="macro", zero_division=0)),
        "f1_lu_train": float(mean_f1_loaded_unloaded(load_train, pred_tr)),
        "confusion_train": confusion_matrix(load_train, pred_tr, labels=LABEL_ORDER),
        "acc": float(accuracy_score(load_test, pred_te)),
        "f1_macro": float(f1_score(load_test, pred_te, average="macro", zero_division=0)),
        "f1_lu": float(mean_f1_loaded_unloaded(load_test, pred_te)),
        "confusion": confusion_matrix(load_test, pred_te, labels=LABEL_ORDER),
    }

def eval_turner_test_metrics(
    y_train_pred: np.ndarray,
    gs_train: np.ndarray,
    load_train: np.ndarray,
    y_test_pred: np.ndarray,
    gs_test: np.ndarray,
    load_test: np.ndarray,
) -> dict:
    """Backward-compatible alias: same as ``eval_turner_train_test_metrics`` (includes train + test keys)."""
    return eval_turner_train_test_metrics(
        y_train_pred, gs_train, load_train, y_test_pred, gs_test, load_test
    )

def class_metrics_turner(y_train_pred, gs_train, load_train, y_test_pred, gs_test, load_test, label: str):
    """Tune δ on train predictions; print train (fit) and test (holdout) Turner classification."""
    ev = eval_turner_train_test_metrics(
        y_train_pred, gs_train, load_train, y_test_pred, gs_test, load_test
    )
    dpos, dneg = ev["dpos"], ev["dneg"]
    print(
        f"  [{label}] train Acc={ev['acc_train']:.4f}  macro-F1={ev['f1_macro_train']:.4f}  "
        f"mean-F1(L,U)={ev['f1_lu_train']:.4f}  (same δ as below; resubstitution on train)"
    )
    print(
        f"  [{label}] test  Acc={ev['acc']:.4f}  macro-F1={ev['f1_macro']:.4f}  mean-F1(L,U)={ev['f1_lu']:.4f}  "
        f"(δ_ref+={dpos:.1f}, δ_ref-={dneg:.1f})"
    )
    print("  Confusion train [rows=true L,Near,U]:\n", ev["confusion_train"])
    print("  Confusion test  [rows=true L,Near,U]:\n", ev["confusion"])
    return dpos, dneg

def fmt_mean_std(a: list[float]) -> str:
    x = np.asarray(a, dtype=float)
    return f"{float(np.mean(x)):,.4f} ± {float(np.std(x)):,.4f}"

def plot_turner_qmeas_vs_qrate(
    q_meas: np.ndarray,
    q_rate: np.ndarray,
    is_train: np.ndarray,
    dpos_ref: float,
    dneg_ref: float,
    title: str,
    y_label: str,
    out_base: Path | None,
    loading: np.ndarray | None = None,
    x_floor: float = 1e4,
    show: bool = False,
) -> None:
    """
    Turner-style cross-plot: x = measured gas rate Q_meas, y = predicted rate (gas or Q_cr).
    Shaded zones use δ(Q)=max(δ_ref, α·Q) for Near / Loaded / Unloaded interpretation.
    """
    q_meas = np.asarray(q_meas, dtype=float).ravel()
    q_rate = np.asarray(q_rate, dtype=float).ravel()
    is_train = np.asarray(is_train, dtype=bool).ravel()
    if loading is not None:
        loading = np.asarray(loading, dtype=float).ravel()
    if not (len(q_meas) == len(q_rate) == len(is_train)):
        raise ValueError(
            f"Length mismatch: q_meas {len(q_meas)}, q_rate {len(q_rate)}, is_train {len(is_train)}"
        )
    if loading is not None and len(loading) != len(q_meas):
        raise ValueError(f"Length mismatch: loading {len(loading)} vs q_meas {len(q_meas)}")
    fin = np.isfinite(q_meas) & np.isfinite(q_rate)
    if not np.any(fin):
        print("  Skipping plot_turner_qmeas_vs_qrate: no finite Q_meas / Q_rate pairs.")
        return
    q_meas = q_meas[fin]
    q_rate = q_rate[fin]
    is_train = is_train[fin]
    if loading is not None:
        loading = loading[fin]

    x_min = max(float(np.nanpercentile(q_meas, 0.5)) * 0.98, x_floor)
    x_max = float(np.nanpercentile(q_meas, 99.5)) * 1.05
    y_min = max(0.0, float(np.nanpercentile(q_rate, 0.5)) * 0.98)
    y_max_raw = float(np.nanpercentile(q_rate, 99.5)) * 1.05
    q_meas_hi = float(np.nanpercentile(q_meas, 99.5))
    q_rate_p99 = float(np.nanpercentile(q_rate, 99.0))
    y_max_cap = max(q_meas_hi * 2.8, q_rate_p99 * 1.25)
    y_max = min(y_max_raw, y_max_cap)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_max = x_min + max(abs(x_min) * 1e-6, 1.0)

    xs = np.linspace(x_min, x_max, 500)
    dpx = np.maximum(dpos_ref, DYN_DELTA_POS_FRAC * xs)
    dnx = np.maximum(dneg_ref, DYN_DELTA_NEG_FRAC * xs)
    lo = np.clip(xs - dnx, y_min, y_max)
    hi = np.clip(xs + dpx, y_min, y_max)

    fig, ax = plt.subplots(figsize=(7.2, 6.5), dpi=300)
    ax.fill_between(xs, lo, hi, color="#E8B87D", alpha=0.22, zorder=0, linewidth=0)
    ax.fill_between(xs, hi, y_max, where=hi < y_max, color="#C44E52", alpha=0.12, zorder=0)
    ax.fill_between(xs, y_min, lo, where=lo > y_min, color="#4C72B0", alpha=0.12, zorder=0)

    cmap = {1: "#C44E52", 0: "#E8B87D", -1: "#4C72B0"}
    if loading is not None and len(loading) == len(q_meas):
        for k, color in cmap.items():
            m = loading == k
            if not np.any(m):
                continue
            ax.scatter(
                q_meas[m & is_train],
                q_rate[m & is_train],
                c=color,
                s=78,
                alpha=0.9,
                edgecolors="0.25",
                linewidths=0.5,
                zorder=5,
            )
            ax.scatter(
                q_meas[m & ~is_train],
                q_rate[m & ~is_train],
                c=color,
                s=84,
                alpha=0.92,
                edgecolors="w",
                linewidths=0.55,
                zorder=6,
            )
    else:
        ax.scatter(
            q_meas[is_train],
            q_rate[is_train],
            c="0.45",
            s=78,
            alpha=0.9,
            edgecolors="0.2",
            linewidths=0.5,
            zorder=5,
            label="Train",
        )
        ax.scatter(
            q_meas[~is_train],
            q_rate[~is_train],
            c="0.35",
            s=84,
            alpha=0.92,
            edgecolors="w",
            linewidths=0.55,
            zorder=6,
            label="Test",
        )

    ax.plot(xs, xs + dpx, ":", color="0.35", lw=1.2, zorder=3)
    ax.plot(xs, np.maximum(xs - dnx, 0.0), ":", color="0.35", lw=1.2, zorder=3)
    a = max(x_min, y_min)
    b = min(x_max, y_max)
    if b > a:
        ax.plot([a, b], [a, b], "k--", lw=1.85, zorder=4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Measured gas flow rate $Q_{\mathrm{meas}}$ (m³/day)", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, ls="--", alpha=0.4)
    handles = [
        Line2D([0], [0], ls=":", color="0.35", lw=1.2, label=rf"Near strip $\delta^{{+}}(Q)$, $\delta^{{-}}(Q)$"),
        Line2D([0], [0], color="k", ls="--", lw=1.85, label=r"1:1 ($Q_{\mathrm{rate}} = Q_{\mathrm{meas}}$)"),
    ]
    if loading is None:
        handles += [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="0.45", markersize=8, label="Train"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="0.35", markeredgecolor="w", markersize=8, label="Test"),
        ]
    else:
        handles += [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[1], markersize=8, label="Loaded"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[0], markersize=8, label="Near L.U"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[-1], markersize=8, label="Unloaded"),
        ]
    ax.legend(handles=handles, loc="best", fontsize=9, frameon=True)
    fig.tight_layout()

    if out_base is not None:
        out_base = Path(out_base)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            pth = out_base.with_suffix(ext)
            fig.savefig(pth, dpi=300, bbox_inches="tight")
            print(f"  Saved plot: {pth}")
    if show:
        plt.show()
    plt.close(fig)

def plot_residual_qminus_qmeas(
    q_meas: np.ndarray,
    q_rate: np.ndarray,
    is_train: np.ndarray,
    dpos_ref: float,
    dneg_ref: float,
    title: str,
    out_base: Path | None,
    loading: np.ndarray | None = None,
    show: bool = False,
) -> None:
    """Residuals: (Q_rate - Q_meas) vs Q_meas with dynamic δ bands."""
    q_meas = np.asarray(q_meas, dtype=float).ravel()
    q_rate = np.asarray(q_rate, dtype=float).ravel()
    is_train = np.asarray(is_train, dtype=bool).ravel()
    if not (len(q_meas) == len(q_rate) == len(is_train)):
        raise ValueError(
            f"Length mismatch: q_meas {len(q_meas)}, q_rate {len(q_rate)}, is_train {len(is_train)}"
        )
    if loading is not None:
        loading = np.asarray(loading, dtype=float).ravel()
        if len(loading) != len(q_meas):
            raise ValueError(f"loading length {len(loading)} != q_meas {len(q_meas)}")
    fin = np.isfinite(q_meas) & np.isfinite(q_rate)
    if not np.any(fin):
        print("  Skipping plot_residual_qminus_qmeas: no finite Q_meas / Q_rate pairs.")
        return
    q_meas = q_meas[fin]
    q_rate = q_rate[fin]
    is_train = is_train[fin]
    diff = q_rate - q_meas
    if loading is not None:
        loading = loading[fin]
    qm_lo = float(np.min(q_meas))
    qm_hi = float(np.max(q_meas))
    if qm_hi <= qm_lo:
        qm_hi = qm_lo + max(abs(qm_lo) * 1e-6, 1.0)
    xs = np.linspace(qm_lo, qm_hi, max(400, len(q_meas)))
    dp = np.maximum(dpos_ref, DYN_DELTA_POS_FRAC * xs)
    dn = np.maximum(dneg_ref, DYN_DELTA_NEG_FRAC * xs)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    ax.axhline(0.0, color="k", ls="--", lw=1.1)
    ax.plot(xs, dp, ":", color="0.45", lw=1.1)
    ax.plot(xs, -dn, ":", color="0.45", lw=1.1)
    ax.fill_between(xs, -dn, dp, color="#E8B87D", alpha=0.2, zorder=0)

    cmap = {1: "#C44E52", 0: "#E8B87D", -1: "#4C72B0"}
    if loading is not None and len(loading) == len(q_meas):
        for k, color in cmap.items():
            m = loading == k
            ax.scatter(q_meas[m & is_train], diff[m & is_train], c=color, s=68, alpha=0.88, edgecolors="0.22", lw=0.45, zorder=3)
            ax.scatter(q_meas[m & ~is_train], diff[m & ~is_train], c=color, s=74, alpha=0.9, edgecolors="w", lw=0.5, zorder=4)
    else:
        ax.scatter(q_meas[is_train], diff[is_train], c="0.4", s=68, alpha=0.88, edgecolors="0.2", lw=0.45, zorder=3)
        ax.scatter(q_meas[~is_train], diff[~is_train], c="0.35", s=74, alpha=0.9, edgecolors="w", lw=0.5, zorder=4)

    ax.set_xlabel(r"$Q_{\mathrm{meas}}$ (m³/day)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$Q_{\mathrm{rate}} - Q_{\mathrm{meas}}$ (m³/day)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    if out_base is not None:
        out_base = Path(out_base)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            pth = Path(str(out_base) + "_residual" + ext)
            fig.savefig(pth, dpi=300, bbox_inches="tight")
            print(f"  Saved plot: {pth}")
    if show:
        plt.show()
    plt.close(fig)


def plot_qcr_x_qmeas_y_multimethod(
    q_meas: np.ndarray,
    q_cr_by_method: dict[str, np.ndarray],
    *,
    loading: np.ndarray | None = None,
    is_train: np.ndarray | None = None,
    out_base: Path | None = None,
    title: str = r"$Q_{\mathrm{meas}}$ vs predicted $Q_{\mathrm{cr}}$ (multiple methods)",
    show: bool = False,
    shade_loading_on_qcr: bool = True,
    draw_one_to_one: bool = True,
    trend_lines: bool = True,
    highlight_near_diagonal: bool = True,
    near_eps_frac: float = 0.02,
) -> None:
    """
    Cross-plot with **x = predicted** :math:`Q_{\\mathrm{cr}}` **per method**, **y = measured**
    :math:`Q_{\\mathrm{meas}}` (same wells, aligned rows).

    * Each method is drawn as a scatter series (and optional linear trend in this plane).
    * Optional light **vertical** bands along the x-axis: per loading class, span the
      (p5–p95) range of :math:`Q_{\\mathrm{cr}}` using the **first** method’s predictions
      (rough visual grouping by status).
    * Optional **1:1** line :math:`Q_{\\mathrm{meas}} = Q_{\\mathrm{cr}}` and ring highlights
      where points lie near that line for each method (intersection with neutrality, not a
      statistical test).
    """
    if not q_cr_by_method:
        print("  Skipping plot_qcr_x_qmeas_y_multimethod: no methods provided.")
        return

    q_meas = np.asarray(q_meas, dtype=float).ravel()
    qcr_arrays = {k: np.asarray(v, dtype=float).ravel() for k, v in q_cr_by_method.items()}
    n_expect = len(q_meas)
    for name, arr in qcr_arrays.items():
        if len(arr) != n_expect:
            raise ValueError(f"Method {name!r}: length {len(arr)} != q_meas {n_expect}")

    if is_train is not None:
        is_train = np.asarray(is_train, dtype=bool).ravel()
        if len(is_train) != n_expect:
            raise ValueError(f"is_train length {len(is_train)} != q_meas {n_expect}")
    if loading is not None:
        loading = np.asarray(loading, dtype=float).ravel()
        if len(loading) != n_expect:
            raise ValueError(f"loading length {len(loading)} != q_meas {n_expect}")

    fin = np.isfinite(q_meas)
    for arr in qcr_arrays.values():
        fin &= np.isfinite(arr)
    if not np.any(fin):
        print("  Skipping plot_qcr_x_qmeas_y_multimethod: no finite pairs.")
        return

    q_meas = q_meas[fin]
    qcr_arrays = {k: v[fin] for k, v in qcr_arrays.items()}
    if is_train is not None:
        is_train = is_train[fin]
    if loading is not None:
        loading = loading[fin]

    # Axis limits from pooled Q_cr and Q_meas
    all_x = np.concatenate(list(qcr_arrays.values()))
    x_min = float(np.nanpercentile(all_x, 1.0))
    x_max = float(np.nanpercentile(all_x, 99.0))
    y_min = max(0.0, float(np.nanpercentile(q_meas, 1.0)))
    y_max = float(np.nanpercentile(q_meas, 99.0))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_max = x_min + max(abs(x_min) * 1e-6, 1.0)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        y_max = y_min + max(abs(y_min) * 1e-6, 1.0)

    fig, ax = plt.subplots(figsize=(7.4, 6.6), dpi=300)
    cmap_bg = {1: "#C44E52", 0: "#E8B87D", -1: "#4C72B0"}
    q_cr_ref = qcr_arrays[next(iter(qcr_arrays.keys()))]

    if shade_loading_on_qcr and loading is not None:
        for k, color in cmap_bg.items():
            m = loading == k
            if not np.any(m):
                continue
            xr = q_cr_ref[m]
            if xr.size < 2:
                continue
            v0, v1 = np.percentile(xr, [5.0, 95.0])
            if v1 <= v0:
                continue
            ax.axvspan(v0, v1, color=color, alpha=0.09, zorder=0, linewidth=0)

    method_colors = ["#C44E52", "#55A868", "#4C72B0", "#8172B2", "#CCB974", "#64B5CD"]
    for i, (name, qcr) in enumerate(qcr_arrays.items()):
        c = method_colors[i % len(method_colors)]
        if is_train is None:
            ax.scatter(
                qcr,
                q_meas,
                c=c,
                s=64,
                alpha=0.85,
                edgecolors="0.25",
                linewidths=0.45,
                label=name,
                zorder=3,
            )
        else:
            ax.scatter(
                qcr[is_train],
                q_meas[is_train],
                c=c,
                s=56,
                alpha=0.82,
                edgecolors="0.25",
                linewidths=0.4,
                marker="o",
                label=f"{name} (train)",
                zorder=3,
            )
            ax.scatter(
                qcr[~is_train],
                q_meas[~is_train],
                c=c,
                s=78,
                alpha=0.9,
                edgecolors="w",
                linewidths=0.55,
                marker="s",
                label=f"{name} (test)",
                zorder=4,
            )

        if trend_lines and len(qcr) >= 2:
            vc = float(np.nanvar(qcr))
            if vc > 1e-12:
                coef = np.polyfit(qcr, q_meas, 1)
                xg = np.linspace(x_min, x_max, 120)
                ax.plot(xg, np.poly1d(coef)(xg), color=c, ls="--", lw=1.35, alpha=0.9, zorder=2)

        if highlight_near_diagonal:
            scale = max(float(np.nanmedian(np.abs(q_meas))), 1.0)
            eps = float(near_eps_frac) * scale
            near = np.abs(q_meas - qcr) <= eps
            if np.any(near):
                ax.scatter(
                    qcr[near],
                    q_meas[near],
                    s=150,
                    facecolors="none",
                    edgecolors=c,
                    linewidths=1.15,
                    zorder=6,
                )

    if draw_one_to_one:
        lo = max(x_min, y_min)
        hi = min(x_max, y_max)
        if hi > lo:
            ax.plot(
                [lo, hi],
                [lo, hi],
                "k-",
                lw=2.0,
                zorder=1,
                label=r"$Q_{\mathrm{meas}} = Q_{\mathrm{cr}}$ (1:1)",
            )

    ax.set_xlabel(r"Predicted $Q_{\mathrm{cr}}$ (m³/day)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Measured gas rate $Q_{\mathrm{meas}}$ (m³/day)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, ls="--", alpha=0.38)

    extra_artists = []
    if shade_loading_on_qcr and loading is not None:
        for k, color in cmap_bg.items():
            if np.any(loading == k):
                extra_artists.append(
                    Line2D([0], [0], color=color, lw=6, alpha=0.35, label=f"Status {k} (Q_cr p5–p95)")
                )
    if highlight_near_diagonal:
        extra_artists.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="0.35",
                markersize=8,
                label="Near 1:1 rings (|Δ| ≤ ε)",
            )
        )
    h, lab = ax.get_legend_handles_labels()
    if extra_artists:
        ax.legend(h + extra_artists, lab + [e.get_label() for e in extra_artists], loc="best", fontsize=7, frameon=True)
    else:
        ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()

    if out_base is not None:
        out_base = Path(out_base)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            pth = out_base.with_suffix(ext)
            fig.savefig(pth, dpi=300, bbox_inches="tight")
            print(f"  Saved plot: {pth}")
    if show:
        plt.show()
    plt.close(fig)


def save_pipeline_predictions_csv(
    export_dir: Path,
    *,
    pipeline: str,
    mode: str,
    seed: int,
    q_meas: np.ndarray,
    q_pred: np.ndarray,
    loading: np.ndarray,
    is_train: np.ndarray,
    pred_column: str,
    delta_pos: float | None = None,
    delta_neg: float | None = None,
    tag: str = "",
    source_csv: str | None = None,
) -> Path:
    """
    Save one row per well (train rows first, then test) with measured gas rate and model prediction.

    Columns include ``split``, ``loading`` (-1/0/1), ``Test status`` (same labels as the well CSV:
    Unloaded / Near L.U / Loaded), ``Q_meas_m3day``, the prediction column
    (e.g. ``Q_cr_pred_m3day`` or ``Q_gas_pred_m3day``), and optional ``delta_*_ref_m3day`` for Turner
    rule replication. Use ``tag`` for e.g. ``cv_last_fold``.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pipeline}_{mode}_seed{int(seed)}"
    if tag:
        stem += f"_{tag}"
    path = export_dir / f"{stem}.csv"
    qm = np.asarray(q_meas, dtype=float).ravel()
    qp = np.asarray(q_pred, dtype=float).ravel()
    ld = np.asarray(loading).ravel()
    it = np.asarray(is_train, dtype=bool).ravel()
    n = len(qm)
    if not (len(qp) == len(ld) == len(it) == n):
        raise ValueError("Length mismatch writing predictions CSV.")
    # Align with processed_well_data.csv "Test status" strings
    _status_map = {-1: "Unloaded", 0: "Near L.U", 1: "Loaded"}

    def _test_status_str(code: float) -> str:
        k = int(round(float(code)))
        return _status_map.get(k, f"unknown({k})")

    test_status = np.array([_test_status_str(x) for x in ld], dtype=object)
    df = pd.DataFrame(
        {
            "split": np.where(it, "train", "test"),
            "is_train": it.astype(np.int8),
            "loading": ld.astype(np.int8),
            "Test status": test_status,
            "Q_meas_m3day": qm,
            pred_column: qp,
        }
    )
    if delta_pos is not None and delta_neg is not None:
        df["delta_pos_ref_m3day"] = float(delta_pos)
        df["delta_neg_ref_m3day"] = float(delta_neg)
    if source_csv:
        df["source_csv"] = str(Path(source_csv).resolve())
    df.to_csv(path, index=False)
    print(f"  Saved predictions CSV: {path}")
    return path


def load_filtered_unscaled(csv_path: Path) -> dict:
    """
    Same filtering as NewTestOpti / single split, but **unscaled** features for k-fold
    (each fold fits its own StandardScaler on the training fold).
    """
    df = pd.read_csv(csv_path)
    df = df[df["Test status"] != "Questionable"].copy()

    feature_cols = list(TURNER_FEATURE_COLS)

    X = df[feature_cols].values.astype(float)
    gsflow = df["Gasflowrate"].values.astype(float)
    loading = df["Test status"].map({"Loaded": 1, "Near L.U": 0, "Unloaded": -1}).values

    flow_bins = pd.qcut(gsflow, q=3, labels=False, duplicates="drop")
    stratify_labels = pd.Series(loading).astype(str).str.cat(
        pd.Series(flow_bins).astype(str), sep="_"
    )
    counts = stratify_labels.value_counts()
    valid_classes = counts[counts >= 2].index
    mask = stratify_labels.isin(valid_classes)

    X = X[mask]
    gsflow = gsflow[mask]
    loading = loading[mask]
    stratify_labels = stratify_labels[mask].values

    _load_counts = pd.Series(loading).value_counts()
    _stratify = loading if _load_counts.min() >= 2 else stratify_labels

    return {
        "feature_cols": feature_cols,
        "X": X,
        "gsflow": gsflow,
        "loading": loading,
        "stratify": _stratify,
    }

def split_raw_train_test(
    raw: dict,
    random_state: int,
    test_size: float = 0.2,
) -> dict:
    """Single train/test split with scaler fit on **all** filtered rows (matches original notebook)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw["X"])
    (
        X_train,
        X_test,
        gs_train,
        gs_test,
        load_train,
        load_test,
    ) = train_test_split(
        X_scaled,
        raw["gsflow"],
        raw["loading"],
        test_size=test_size,
        random_state=random_state,
        stratify=raw["stratify"],
    )

    med = float(np.nanmedian(gs_train))
    set_loss_margins(med)
    print(
        f"Loaded {len(X_train)} train / {len(X_test)} test wells | "
        f"median train gas {med:,.0f} | LOSS_DELTA L/U/N = "
        f"{LOSS_DELTA_LOADED:,.0f} / {LOSS_DELTA_UNLOADED:,.0f} / {LOSS_DELTA_NEAR:,.0f}"
    )

    return {
        "feature_cols": raw["feature_cols"],
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "gs_train": gs_train,
        "gs_test": gs_test,
        "load_train": load_train,
        "load_test": load_test,
    }

def load_and_split(
    csv_path: Path,
    random_state: int,
    test_size: float = 0.2,
):
    raw = load_filtered_unscaled(csv_path)
    return split_raw_train_test(raw, random_state, test_size)

def _gas_target_forward(gs: np.ndarray, log1p_gas: bool) -> np.ndarray:
    g = np.asarray(gs, dtype=float)
    if log1p_gas:
        return np.log1p(np.maximum(g, 0.0))
    return g

def _gas_pred_inverse(pred: np.ndarray, log1p_gas: bool) -> np.ndarray:
    """Map Feyn outputs back to physical gas rate (m³/day)."""
    p = np.asarray(pred, dtype=float)
    if log1p_gas:
        return np.maximum(0.0, np.expm1(np.clip(p, -50.0, 50.0)))
    return p

def q_critical_label_proxy_targets(
    gs: np.ndarray,
    load: np.ndarray,
    *,
    margin_scale: float = 1.0,
) -> np.ndarray:
    """
    Scalar training targets for symbolic Q_critical **without** any other model.
    Uses the same LOSS_DELTA_* globals as the notebook (set by load_and_split):
      Loaded   -> Q_meas + margin_scale * LOSS_DELTA_LOADED
      Unloaded -> Q_meas - margin_scale * LOSS_DELTA_UNLOADED
      Near L.U -> Q_meas (center of near band)
    True Q_critical is still unknown; this is only a simple label-shaped target for Feyn.
    ``margin_scale`` scales only the L/U offsets (e.g. >1 pushes training targets farther from Q_meas);
    Turner evaluation bands are unchanged.
    """
    ms = float(margin_scale)
    g = np.asarray(gs, dtype=float)
    lab = np.asarray(load, dtype=float)
    t = g.copy()
    t[lab == 1] += ms * float(LOSS_DELTA_LOADED)
    t[lab == -1] -= ms * float(LOSS_DELTA_UNLOADED)
    return t

def print_mean_residual_by_loading(
    pred: np.ndarray,
    q_meas: np.ndarray,
    load: np.ndarray,
    header: str,
) -> None:
    """Diagnostic: class-wise mean(Q_pred - Q_meas); useful when Turner cross-plot looks tight on the diagonal."""
    pred = np.asarray(pred, dtype=float).ravel()
    q_meas = np.asarray(q_meas, dtype=float).ravel()
    load = np.asarray(load, dtype=float).ravel()
    for name, val in [("Loaded", 1.0), ("Near L.U", 0.0), ("Unloaded", -1.0)]:
        m = load == val
        if np.any(m):
            r = float(np.mean(pred[m] - q_meas[m]))
            print(f"  [{header}] mean(Q_pred - Q_meas) | {name}: {r:,.1f} m³/day  (n={int(m.sum())})")

def append_measured_gas_feature(
    train_X_df: pd.DataFrame,
    test_X_df: pd.DataFrame,
    gs_train: np.ndarray,
    gs_test: np.ndarray,
    enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append measured gas rate as an extra column for Q_critical Feyn inputs (optional; strong signal / leakage)."""
    if not enabled:
        return train_X_df, test_X_df
    tr = train_X_df.copy()
    te = test_X_df.copy()
    gtr = np.asarray(gs_train, dtype=float).ravel()
    gte = np.asarray(gs_test, dtype=float).ravel()
    tr["Gasflowrate_meas"] = gtr
    te["Gasflowrate_meas"] = gte
    return tr, te

def feyn_run_qcritical_fit(
    train_X_df: pd.DataFrame,
    test_X_df: pd.DataFrame,
    gs_train: np.ndarray,
    load_train: np.ndarray,
    y_train_target: np.ndarray,
    args: argparse.Namespace,
    feyn_mod: object,
    ql_seed: int,
) -> tuple[object, np.ndarray, np.ndarray, bool, int, float]:
    """Shared Feyn auto_run + optional val model pick for Q_critical regression."""
    inner_idx_q, val_idx_q = train_val_indices(
        gs_train,
        load_train,
        args.val_frac,
        ql_seed + 19,
    )
    use_val_q = (not args.no_val_select) and val_idx_q.size > 0
    if use_val_q:
        train_q = train_X_df.iloc[inner_idx_q].copy()
        train_q["Q_critical"] = y_train_target[inner_idx_q]
        print(
            f"  Q_critical fit: {len(inner_idx_q)} train inner / {len(val_idx_q)} val for model selection "
            f"({args.select_by!r})."
        )
    else:
        train_q = train_X_df.copy()
        train_q["Q_critical"] = y_train_target
        if not args.no_val_select:
            print("  (Train set too small for val split — using full train and first auto_run model.)")
        else:
            print("  (--no-val-select: full train, first auto_run model.)")

    ql_q = feyn_mod.QLattice(random_seed=ql_seed)
    models_q = feyn_auto_run_safe(
        ql_q,
        data=train_q,
        output_name="Q_critical",
        kind="regression",
        n_epochs=args.epochs,
        max_complexity=args.max_complexity,
        loss_function=None if args.loss == "squared_error" else args.loss,
    )
    if use_val_q:
        best_q, pick_jq, val_score_q = pick_best_feyn_model(
            models_q,
            train_X_df,
            gs_train,
            load_train,
            inner_idx_q,
            val_idx_q,
            args.select_by,
            args.top_models,
            fit_target=y_train_target,
        )
        print(
            f"  Selected Q_critical model #{pick_jq + 1} / {min(args.top_models, len(models_q))}  "
            f"val score={val_score_q:.4f}"
        )
    else:
        best_q = models_q[0]
        pick_jq, val_score_q = 0, float("nan")
        print(f"  Using auto_run rank-1 Q_critical model (of {len(models_q)} returned).")

    pred_tr_q = np.asarray(best_q.predict(train_X_df), dtype=float).ravel()
    pred_te_q = np.asarray(best_q.predict(test_X_df), dtype=float).ravel()
    return best_q, pred_tr_q, pred_te_q, use_val_q, pick_jq, val_score_q

def load_qcr_train_test_from_npz(z: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """Physics-model Q_critical per row (m3/day); true Q_critical is unknown; targets for symbolic fit."""
    names = set(z.files)
    if "y_train_pred" in names and "y_test_pred" in names:
        return np.asarray(z["y_train_pred"]).ravel(), np.asarray(z["y_test_pred"]).ravel()
    if "q_critical_train" in names and "q_critical_test" in names:
        return np.asarray(z["q_critical_train"]).ravel(), np.asarray(z["q_critical_test"]).ravel()
    raise SystemExit(
        f"{names}: need (y_train_pred, y_test_pred) or (q_critical_train, q_critical_test) "
        "with physics-model Q_cr estimates aligned to this script's train/test split."
    )

def fold_train_test_frames(
    raw: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-fold scaler (fit on train only) → feature DataFrames in physical units."""
    fc = raw["feature_cols"]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(raw["X"][train_idx])
    X_te = scaler.transform(raw["X"][test_idx])
    train_X_df = pd.DataFrame(scaler.inverse_transform(X_tr), columns=fc)
    test_X_df = pd.DataFrame(scaler.inverse_transform(X_te), columns=fc)
    gs_train = raw["gsflow"][train_idx]
    gs_test = raw["gsflow"][test_idx]
    load_train = raw["loading"][train_idx]
    load_test = raw["loading"][test_idx]
    return train_X_df, test_X_df, gs_train, gs_test, load_train, load_test

def make_cv_folds(stratify_y: np.ndarray, n_splits: int, random_state: int) -> tuple[str, list[tuple[np.ndarray, np.ndarray]], int]:
    """
    Prefer StratifiedKFold on loading-like labels; fall back to shuffled KFold if needed.
    Returns (kind, list of (train_idx, test_idx), n_splits_used).
    """
    n = len(stratify_y)
    n_splits = int(max(2, min(n_splits, n)))
    if n_splits > n // 2:
        n_splits = max(2, n // 2)
    rng_state = random_state

    while n_splits >= 2:
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_state)
            folds = list(skf.split(np.zeros(n), stratify_y))
            return "stratified", folds, n_splits
        except ValueError:
            n_splits -= 1

    kf = KFold(n_splits=max(2, min(5, n // 3)), shuffle=True, random_state=rng_state)
    folds = list(kf.split(np.zeros(n)))
    return "kfold", folds, len(folds)

def train_predict_gas_qlattice(
    train_X_df: pd.DataFrame,
    test_X_df: pd.DataFrame,
    gs_train: np.ndarray,
    gs_test: np.ndarray,
    load_train: np.ndarray,
    load_test: np.ndarray,
    args: argparse.Namespace,
    feyn_module: object,
    ql_seed: int,
    *,
    verbose: bool,
) -> dict:
    """Fit QLattice on gas rate, optional inner val model pick, return metrics and predictions."""
    log1p_gas = not bool(getattr(args, "linear_gas", False))
    inner_idx, val_idx = train_val_indices(gs_train, load_train, args.val_frac, ql_seed + 11)
    use_val = (not args.no_val_select) and val_idx.size > 0
    if use_val:
        train_g = train_X_df.iloc[inner_idx].copy()
        train_g["Gasflowrate"] = _gas_target_forward(gs_train[inner_idx], log1p_gas)
        if verbose:
            tgt = "log1p(Gasflowrate)" if log1p_gas else "Gasflowrate"
            print(
                f"  Training on {len(inner_idx)} wells; validating on {len(val_idx)} to select among "
                f"top-{args.top_models} models by {args.select_by!r}  [target={tgt}]."
            )
    else:
        train_g = train_X_df.copy()
        train_g["Gasflowrate"] = _gas_target_forward(gs_train, log1p_gas)
        if verbose:
            tgt = "log1p(Gasflowrate)" if log1p_gas else "Gasflowrate"
            if not args.no_val_select:
                print(f"  (Train set too small for val split — using full train and first auto_run model.)  [target={tgt}]")
            else:
                print(f"  (--no-val-select: full train, first auto_run model.)  [target={tgt}]")

    ql_g = feyn_module.QLattice(random_seed=ql_seed)
    models_g = feyn_auto_run_safe(
        ql_g,
        data=train_g,
        output_name="Gasflowrate",
        kind="regression",
        n_epochs=args.epochs,
        max_complexity=args.max_complexity,
        loss_function=None if args.loss == "squared_error" else args.loss,
    )
    if use_val:
        best_g, pick_j, val_score = pick_best_feyn_model(
            models_g,
            train_X_df,
            gs_train,
            load_train,
            inner_idx,
            val_idx,
            args.select_by,
            args.top_models,
            log1p_gas=log1p_gas,
        )
        if verbose:
            print(
                f"  Selected model #{pick_j + 1} / {min(args.top_models, len(models_g))}  "
                f"val score={val_score:.4f}"
            )
    else:
        best_g = models_g[0]
        pick_j, val_score = 0, float("nan")
        if verbose:
            print(f"  Using auto_run rank-1 model (of {len(models_g)} returned).")

    pred_tr = _gas_pred_inverse(np.asarray(best_g.predict(train_X_df), dtype=float).ravel(), log1p_gas)
    pred_te = _gas_pred_inverse(np.asarray(best_g.predict(test_X_df), dtype=float).ravel(), log1p_gas)
    mae = float(np.mean(np.abs(pred_te - gs_test)))
    rmse = float(np.sqrt(np.mean((pred_te - gs_test) ** 2)))
    ev = eval_turner_test_metrics(pred_tr, gs_train, load_train, pred_te, gs_test, load_test)
    return {
        "best_g": best_g,
        "mae": mae,
        "rmse": rmse,
        "pred_tr": pred_tr,
        "pred_te": pred_te,
        "pick_j": pick_j,
        "val_score": val_score,
        **ev,
    }

def _print_model_eq(m, title: str, *, note: str | None = None) -> None:
    print(f"\n--- {title} (best model summary) ---")
    if note:
        print(f"  {note}")
    try:
        sp = m.sympify(signif=4)
        print(sp)
    except Exception as exc:
        print("(sympify unavailable)", exc)
    try:
        print(m)
    except Exception:
        pass

def train_val_indices(
    gs_train: np.ndarray,
    load_train: np.ndarray,
    val_frac: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Indices into the training set for (inner fit, validation) used to pick among auto_run models.
    Falls back through stratify options; last resort is a random split.
    """
    n = len(load_train)
    idx = np.arange(n)
    vf = float(val_frac)
    if vf <= 0 or vf >= 0.5:
        vf = 0.2
    if n < 16:
        return np.sort(idx), np.array([], dtype=int)

    flow_bins = pd.qcut(gs_train, q=3, labels=False, duplicates="drop")
    joint = pd.Series(load_train).astype(str).str.cat(pd.Series(flow_bins).astype(str), sep="_")
    counts = joint.value_counts()
    joint_strat = joint.where(joint.isin(counts[counts >= 2].index))

    candidates: list = []
    if int(pd.Series(load_train).value_counts().min()) >= 2:
        candidates.append(load_train)
    if joint_strat.notna().all() and joint_strat.nunique() >= 2:
        candidates.append(joint_strat)
    candidates.append(None)

    last_err: Exception | None = None
    for strat in candidates:
        try:
            tr, va = train_test_split(
                idx,
                test_size=vf,
                random_state=random_state,
                stratify=strat,
            )
            if len(va) < 2 or len(tr) < 8:
                continue
            return np.sort(tr), np.sort(va)
        except ValueError as e:
            last_err = e
            continue
    tr, va = train_test_split(idx, test_size=vf, random_state=random_state)
    return np.sort(tr), np.sort(va)

def _turner_val_score(
    pred_inner: np.ndarray,
    gs_inner: np.ndarray,
    load_inner: np.ndarray,
    pred_val: np.ndarray,
    gs_val: np.ndarray,
    load_val: np.ndarray,
    metric: str,
) -> float:
    dpos, dneg = tune_delta(pred_inner, gs_inner, load_inner)
    pred_cls = predict_status(pred_val, gs_val, dpos, dneg)
    if metric == "macro":
        return float(f1_score(load_val, pred_cls, average="macro", zero_division=0))
    return mean_f1_loaded_unloaded(load_val, pred_cls)

def pick_best_feyn_model(
    models: list,
    train_X_df: pd.DataFrame,
    gs_train: np.ndarray,
    load_train: np.ndarray,
    inner_idx: np.ndarray,
    val_idx: np.ndarray,
    select_by: str,
    top_k: int,
    *,
    log1p_gas: bool = False,
    fit_target: np.ndarray | None = None,
) -> tuple[object, int, float]:
    """
    Pick the best of the first ``top_k`` diverse models using validation metrics.

    For **gas** regression, ``fit_target`` is omitted: MAE/hybrid compare predictions to **measured gas**
    on the val set.

    For **Q_critical** regression, pass ``fit_target=y_train_target`` (same length as ``gs_train``) so
    ``mae`` / ``hybrid`` score against the **training targets** (physics teacher or label proxy), not raw
    ``Q_meas``. Turner-based options still use ``Q_pred - Q_meas`` vs δ on val.
    """
    if not val_idx.size:
        return models[0], 0, float("nan")

    train_inner = train_X_df.iloc[inner_idx]
    train_val_df = train_X_df.iloc[val_idx]
    gs_i = gs_train[inner_idx]
    load_i = load_train[inner_idx]
    gs_v = gs_train[val_idx]
    load_v = load_train[val_idx]
    reg_v = gs_v if fit_target is None else np.asarray(fit_target, dtype=float)[val_idx]

    k = max(1, min(int(top_k), len(models)))
    best_score = -np.inf
    best_j = 0
    best_detail = float("nan")

    scale_mae = float(np.median(reg_v)) + 1.0

    for j in range(k):
        m = models[j]
        p_i = _gas_pred_inverse(np.asarray(m.predict(train_inner), dtype=float).ravel(), log1p_gas)
        p_v = _gas_pred_inverse(np.asarray(m.predict(train_val_df), dtype=float).ravel(), log1p_gas)

        if select_by == "mae":
            sc = -float(np.mean(np.abs(p_v - reg_v)))
        elif select_by == "turner_lu":
            sc = _turner_val_score(p_i, gs_i, load_i, p_v, gs_v, load_v, "lu")
        elif select_by == "turner_macro":
            sc = _turner_val_score(p_i, gs_i, load_i, p_v, gs_v, load_v, "macro")
        elif select_by == "hybrid":
            mae_norm = -float(np.mean(np.abs(p_v - reg_v))) / scale_mae
            turn = _turner_val_score(p_i, gs_i, load_i, p_v, gs_v, load_v, "lu")
            sc = 0.5 * mae_norm + 0.5 * turn
        else:
            sc = -float(np.mean(np.abs(p_v - reg_v)))

        if sc > best_score:
            best_score = sc
            best_j = j
            best_detail = sc

    return models[best_j], best_j, best_detail

def feyn_auto_run_safe(
    ql: object,
    *,
    data: pd.DataFrame,
    output_name: str,
    kind: str,
    n_epochs: int,
    max_complexity: int,
    loss_function: str | None,
) -> list:
    """Call ``auto_run``, passing ``loss_function`` only when supported by this feyn version."""
    kwargs: dict = {
        "data": data,
        "output_name": output_name,
        "kind": kind,
        "n_epochs": n_epochs,
        "max_complexity": max_complexity,
    }
    if loss_function:
        try:
            return ql.auto_run(**kwargs, loss_function=loss_function)
        except TypeError:
            print(
                f"  (note: this feyn build ignored loss_function={loss_function!r}; using default loss)"
            )
    return ql.auto_run(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone QLattice (feyn) symbolic regression")
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to processed_well_data.csv (default: <repo>/datasets/processed_well_data.csv)",
    )
    p.add_argument(
        "--mode",
        choices=("gas", "qcr_custom", "qcr_labels", "both"),
        default="both",
        help="gas = predict measured rate. qcr_custom = symbolic Q_critical fit to .npz physics-model targets. "
        "qcr_labels = symbolic Q_critical fit to label-implied Q_meas+/-margins (no .npz, no match to other code). "
        "both = gas + qcr_custom (needs .npz for part B).",
    )
    p.add_argument(
        "--qcr-npz",
        type=Path,
        default=None,
        help="Used with qcr_custom/both only: physics-model Q_critical per row (teacher targets): "
        "y_train_pred/y_test_pred or q_critical_train/q_critical_test (m3/day). Ignored for qcr_labels.",
    )
    p.add_argument(
        "--qcr-proxy-margin-scale",
        type=float,
        default=1.0,
        help="qcr_labels only: multiply LOSS_DELTA offsets for loaded/unloaded proxy targets (default 1). "
        ">1 widens training targets away from Q_meas (may reduce 'hugging the 1:1 line'); "
        "Turner delta bands and classification metrics still use tuned delta, not this scale.",
    )
    p.add_argument(
        "--qcr-use-measured-gas",
        action="store_true",
        help="Q_critical modes only: add column Gasflowrate_meas (= measured rate per row) to Feyn inputs. "
        "Usually improves accuracy but uses Q_meas as a feature (not for predicting Q_cr from geometry alone).",
    )
    p.add_argument("--epochs", type=int, default=32, help="QLattice epochs (higher is often better fit, slower)")
    p.add_argument("--max-complexity", type=int, default=12, help="Max graph complexity for symbolic models")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--loss",
        choices=("squared_error", "absolute_error"),
        default="absolute_error",
        help="Regression loss for auto_run (MAE-oriented default; squared_error is classic least-squares).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Hold out this fraction of training rows to score candidate models (Turner / MAE).",
    )
    p.add_argument(
        "--top-models",
        type=int,
        default=15,
        help="Evaluate the first N diverse models from auto_run (Feyn returns a short list sorted by loss).",
    )
    p.add_argument(
        "--select-by",
        choices=("turner_lu", "turner_macro", "mae", "hybrid"),
        default="turner_lu",
        help="Val criterion for picking among auto_run models. turner_* = classify val wells from (Q_cr_pred-Q_meas) vs delta. "
        "mae / hybrid (Q_critical only): compare Q_cr_pred to **training targets** (proxy or .npz), not Q_meas. "
        "hybrid = 0.5*(normalized target MAE) + 0.5*Turner L/U F1. Gas mode: mae/hybrid vs measured rate.",
    )
    p.add_argument(
        "--no-val-select",
        action="store_true",
        help="Train on all training rows and use auto_run's first model (legacy / fastest).",
    )
    p.add_argument(
        "--linear-gas",
        action="store_true",
        help="Regress raw Gasflowrate in Feyn (default: train on log1p(rate), then expm1 for metrics; often fixes high-Q under-prediction).",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If >=2: **gas** mode runs k-fold on gas; **qcr_labels** runs k-fold symbolic Q_critical + Turner per fold "
        "(single 80/20 split when 0). qcr_custom ignores CV (uses one split + .npz).",
    )
    p.add_argument(
        "--cv-plot-last",
        action="store_true",
        help="With --cv-folds, save Turner plots for the **last** fold only (same --plot-dir).",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for Turner cross-plots (PDF+PNG). Default: current working directory.",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip Q_meas vs Q_rate Turner plots")
    p.add_argument("--show-plots", action="store_true", help="Display plots interactively (after saving if --plot-dir set)")
    p.add_argument(
        "--export-csv-dir",
        type=Path,
        default=None,
        help="Write Q_meas + prediction columns to this directory (default: same as --plot-dir).",
    )
    p.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Do not write prediction CSV files (default: export when a run produces predictions).",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.csv is None:
        args.csv = default_processed_well_csv()
    return args


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    try:
        import feyn
    except ImportError:
        raise SystemExit(
            "Missing package: pip install feyn\n"
            "See https://pypi.org/project/feyn/ (community license applies)."
        )

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    data = load_and_split(args.csv, args.seed)
    fc = data["feature_cols"]
    train_X_df = pd.DataFrame(data["scaler"].inverse_transform(data["X_train"]), columns=fc)
    test_X_df = pd.DataFrame(data["scaler"].inverse_transform(data["X_test"]), columns=fc)

    run_gas = args.mode in ("gas", "both")
    run_qcr_custom = args.mode in ("qcr_custom", "both")
    run_qcr_labels = args.mode == "qcr_labels"

    plot_dir = args.plot_dir if args.plot_dir is not None else Path.cwd()
    do_plots = not args.no_plots
    show_plots = bool(args.show_plots)
    export_csv = not args.no_export_csv
    export_dir = args.export_csv_dir if args.export_csv_dir is not None else plot_dir
    csv_src = str(args.csv)

    cv_k = max(0, int(args.cv_folds))
    qcr_labels_ran_cv = False
    raw_for_cv: dict | None = None
    if run_gas and cv_k >= 2:
        raw_for_cv = load_filtered_unscaled(args.csv)

    if run_gas and cv_k >= 2 and raw_for_cv is not None:
        print("\n========== QLattice mode A: Gasflowrate — K-fold cross-validation ==========")
        if run_qcr_custom:
            print(
                "  (Note: k-fold here is for gas only; qcr_custom still uses one train/test split below.)"
            )
        kind, folds, k_eff = make_cv_folds(raw_for_cv["stratify"], cv_k, args.seed)
        print(f"  {k_eff} folds ({kind}), N={len(raw_for_cv['gsflow'])} wells after filtering.")
        if not args.linear_gas:
            print(
                "  log1p gas target: Feyn fits log1p(Q_meas); MAE/Turner/plots use expm1 (often fixes high-Q bias). "
                "Use --linear-gas for raw-rate training."
            )
            print(
                "  For symbolic Q_critical without .npz, use --mode qcr_labels. For physics-teacher fit, use qcr_custom + npz."
            )

        maes: list[float] = []
        rmses: list[float] = []
        accs: list[float] = []
        f1ms: list[float] = []
        f1lus: list[float] = []

        last_out: dict | None = None
        last_frames: tuple | None = None

        for fi, (tr_i, te_i) in enumerate(folds):
            med_fold = float(np.nanmedian(raw_for_cv["gsflow"][tr_i]))
            set_loss_margins(med_fold)
            tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te = fold_train_test_frames(raw_for_cv, tr_i, te_i)
            out = train_predict_gas_qlattice(
                tr_df,
                te_df,
                gs_tr,
                gs_te,
                ld_tr,
                ld_te,
                args,
                feyn,
                args.seed + fi * 97,
                verbose=False,
            )
            maes.append(out["mae"])
            rmses.append(out["rmse"])
            accs.append(out["acc"])
            f1ms.append(out["f1_macro"])
            f1lus.append(out["f1_lu"])
            print(
                f"  Fold {fi + 1}/{k_eff}: MAE={out['mae']:,.2f}  RMSE={out['rmse']:,.2f}  "
                f"Acc={out['acc']:.4f}  macro-F1={out['f1_macro']:.4f}  mean-F1(L,U)={out['f1_lu']:.4f}"
            )
            last_out = out
            last_frames = (tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te)

        print("\n  --- CV summary (test fold) ---")
        print(f"  MAE (m³/day):     {fmt_mean_std(maes)}")
        print(f"  RMSE (m³/day):    {fmt_mean_std(rmses)}")
        print(f"  Acc:              {fmt_mean_std(accs)}")
        print(f"  macro-F1:         {fmt_mean_std(f1ms)}")
        print(f"  mean-F1(L,U):     {fmt_mean_std(f1lus)}")

        if last_out is not None and last_frames is not None:
            tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te = last_frames
            set_loss_margins(float(np.nanmedian(gs_tr)))
            print("\n--- Last fold: Gasflowrate model (reference) ---")
            _gas_sym_note = (
                None
                if args.linear_gas
                else "Feyn fits log1p(Q_gas); printed equation is in that space — metrics/plots use expm1 → m³/day."
            )
            _print_model_eq(last_out["best_g"], "Gasflowrate model", note=_gas_sym_note)

        if export_csv and last_out is not None and last_frames is not None:
            _tr_df, _te_df, gs_tr, gs_te, ld_tr, ld_te = last_frames
            save_pipeline_predictions_csv(
                export_dir,
                pipeline="qlattice",
                mode="gas",
                seed=args.seed,
                q_meas=np.concatenate([gs_tr, gs_te]),
                q_pred=np.concatenate([last_out["pred_tr"], last_out["pred_te"]]),
                loading=np.concatenate([ld_tr, ld_te]),
                is_train=np.concatenate(
                    [np.ones(len(gs_tr), dtype=bool), np.zeros(len(gs_te), dtype=bool)]
                ),
                pred_column="Q_gas_pred_m3day",
                delta_pos=float(last_out["dpos"]),
                delta_neg=float(last_out["dneg"]),
                tag="cv_last_fold",
                source_csv=csv_src,
            )

        if do_plots and args.cv_plot_last and last_out is not None and last_frames is not None:
            tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te = last_frames
            dpos_g = last_out["dpos"]
            dneg_g = last_out["dneg"]
            pred_tr = last_out["pred_tr"]
            pred_te = last_out["pred_te"]
            q_meas_all = np.concatenate([gs_tr, gs_te])
            q_rate_all = np.concatenate([pred_tr, pred_te])
            is_train = np.concatenate(
                [np.ones(len(gs_tr), dtype=bool), np.zeros(len(gs_te), dtype=bool)]
            )
            load_all = np.concatenate([ld_tr, ld_te])
            base = plot_dir / "qlattice_gas_turner_cv_last"
            plot_turner_qmeas_vs_qrate(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_g,
                dneg_g,
                "QLattice gas vs measured — last CV fold (Turner δ bands)",
                r"Predicted gas rate $Q_{\mathrm{pred}}$ (m³/day)",
                base,
                loading=load_all,
                show=show_plots,
            )
            plot_residual_qminus_qmeas(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_g,
                dneg_g,
                "Residual: QLattice gas - Q_meas (last CV fold)",
                base,
                loading=load_all,
                show=show_plots,
            )
    elif run_gas:
        print("\n========== QLattice mode A: Gasflowrate (features → measured rate) ==========")
        if not args.linear_gas:
            print(
                "  log1p gas target: Feyn fits log1p(Q_meas); metrics/plots use expm1. "
                "Use --linear-gas for raw Gasflowrate (previous default behavior)."
            )
        print(
            "  Note: this fits measured rate only. For symbolic Q_critical vs measured flow (Turner), use "
            "--mode qcr_labels (no .npz) or qcr_custom (physics teacher in .npz)."
        )
        out_g = train_predict_gas_qlattice(
            train_X_df,
            test_X_df,
            data["gs_train"],
            data["gs_test"],
            data["load_train"],
            data["load_test"],
            args,
            feyn,
            args.seed,
            verbose=True,
        )
        print(f"  Test MAE (pred vs measured gas): {out_g['mae']:,.2f} m³/day")
        print(f"  Test RMSE: {out_g['rmse']:,.2f} m³/day")
        _gas_sym_note = (
            None
            if args.linear_gas
            else "Feyn fits log1p(Q_gas); printed equation is in that space — metrics/plots use expm1 → m³/day."
        )
        _print_model_eq(out_g["best_g"], "Gasflowrate model", note=_gas_sym_note)
        dpos_g, dneg_g = class_metrics_turner(
            out_g["pred_tr"],
            data["gs_train"],
            data["load_train"],
            out_g["pred_te"],
            data["gs_test"],
            data["load_test"],
            "Turner (QLattice gas predictor — exploratory)",
        )
        if export_csv:
            q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
            q_rate_all = np.concatenate([out_g["pred_tr"], out_g["pred_te"]])
            is_train = np.concatenate(
                [
                    np.ones(len(data["gs_train"]), dtype=bool),
                    np.zeros(len(data["gs_test"]), dtype=bool),
                ]
            )
            load_all = np.concatenate([data["load_train"], data["load_test"]])
            save_pipeline_predictions_csv(
                export_dir,
                pipeline="qlattice",
                mode="gas",
                seed=args.seed,
                q_meas=q_meas_all,
                q_pred=q_rate_all,
                loading=load_all,
                is_train=is_train,
                pred_column="Q_gas_pred_m3day",
                delta_pos=float(out_g["dpos"]),
                delta_neg=float(out_g["dneg"]),
                source_csv=csv_src,
            )
        if do_plots:
            q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
            q_rate_all = np.concatenate([out_g["pred_tr"], out_g["pred_te"]])
            is_train = np.concatenate(
                [
                    np.ones(len(data["gs_train"]), dtype=bool),
                    np.zeros(len(data["gs_test"]), dtype=bool),
                ]
            )
            load_all = np.concatenate([data["load_train"], data["load_test"]])
            base = plot_dir / "qlattice_gas_turner"
            plot_turner_qmeas_vs_qrate(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_g,
                dneg_g,
                "QLattice gas rate vs measured (Turner δ bands)",
                r"Predicted gas rate $Q_{\mathrm{pred}}$ (m³/day)",
                base,
                loading=load_all,
                show=show_plots,
            )
            plot_residual_qminus_qmeas(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_g,
                dneg_g,
                "Residual: QLattice gas - Q_meas",
                base,
                loading=load_all,
                show=show_plots,
            )

    if run_qcr_labels and cv_k >= 2:
        print("\n========== QLattice qcr_labels — K-fold cross-validation ==========")
        print(
            "  Each fold: fit symbolic Q_critical on proxy targets (train split), tune δ on that train split, "
            "report Turner train (resubstitution) vs held-out fold test. "
            "Use --cv-folds 0 for a single 80/20 split with full train+test printout."
        )
        raw_qcr = load_filtered_unscaled(args.csv)
        kind, folds, k_eff = make_cv_folds(raw_qcr["stratify"], cv_k, args.seed)
        print(f"  {k_eff} folds ({kind}), N={len(raw_qcr['gsflow'])} wells after filtering.")
        pscale = float(args.qcr_proxy_margin_scale)
        if pscale <= 0:
            raise SystemExit("--qcr-proxy-margin-scale must be positive.")
        if pscale != 1.0:
            print(f"  Proxy margin scale = {pscale}")
        accs_tr: list[float] = []
        accs_te: list[float] = []
        f1ms: list[float] = []
        f1lus: list[float] = []
        last_qcv: dict | None = None
        for fi, (tr_i, te_i) in enumerate(folds):
            med_fold = float(np.nanmedian(raw_qcr["gsflow"][tr_i]))
            set_loss_margins(med_fold)
            tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te = fold_train_test_frames(raw_qcr, tr_i, te_i)
            tr_df, te_df = append_measured_gas_feature(
                tr_df, te_df, gs_tr, gs_te, bool(getattr(args, "qcr_use_measured_gas", False))
            )
            ytr = q_critical_label_proxy_targets(gs_tr, ld_tr, margin_scale=pscale)
            best_q, pred_tr, pred_te, _, _, _ = feyn_run_qcritical_fit(
                tr_df,
                te_df,
                gs_tr,
                ld_tr,
                ytr,
                args,
                feyn,
                args.seed + 7 + fi * 97,
            )
            ev = eval_turner_train_test_metrics(pred_tr, gs_tr, ld_tr, pred_te, gs_te, ld_te)
            accs_tr.append(ev["acc_train"])
            accs_te.append(ev["acc"])
            f1ms.append(ev["f1_macro"])
            f1lus.append(ev["f1_lu"])
            print(
                f"  Fold {fi + 1}/{k_eff}: train Acc={ev['acc_train']:.4f}  test Acc={ev['acc']:.4f}  "
                f"macro-F1={ev['f1_macro']:.4f}  mean-F1(L,U)={ev['f1_lu']:.4f}  "
                f"(δ+={ev['dpos']:.0f}, δ-={ev['dneg']:.0f})"
            )
            last_qcv = {
                "best_q": best_q,
                "pred_tr": pred_tr,
                "pred_te": pred_te,
                "gs_tr": gs_tr,
                "gs_te": gs_te,
                "ld_tr": ld_tr,
                "ld_te": ld_te,
                "ev": ev,
            }
        print("\n  --- CV summary (Turner; δ tuned per fold on train only) ---")
        print(f"  Train Acc (resubstitution): {fmt_mean_std(accs_tr)}")
        print(f"  Test Acc (held-out fold):   {fmt_mean_std(accs_te)}")
        print(f"  macro-F1 (test):            {fmt_mean_std(f1ms)}")
        print(f"  mean-F1(L,U) (test):        {fmt_mean_std(f1lus)}")
        if last_qcv is not None:
            print("\n--- Last fold: symbolic Q_critical (reference equation) ---")
            _print_model_eq(
                last_qcv["best_q"],
                "Symbolic Q_critical (label-proxy, last CV fold)",
                note="Equation is from the last fold only; margins δ differ per fold.",
            )
        if export_csv and last_qcv is not None:
            ev = last_qcv["ev"]
            gs_tr, gs_te = last_qcv["gs_tr"], last_qcv["gs_te"]
            ld_tr, ld_te = last_qcv["ld_tr"], last_qcv["ld_te"]
            pred_tr, pred_te = last_qcv["pred_tr"], last_qcv["pred_te"]
            save_pipeline_predictions_csv(
                export_dir,
                pipeline="qlattice",
                mode="qcr_labels",
                seed=args.seed,
                q_meas=np.concatenate([gs_tr, gs_te]),
                q_pred=np.concatenate([pred_tr, pred_te]),
                loading=np.concatenate([ld_tr, ld_te]),
                is_train=np.concatenate(
                    [np.ones(len(gs_tr), dtype=bool), np.zeros(len(gs_te), dtype=bool)]
                ),
                pred_column="Q_cr_pred_m3day",
                delta_pos=float(ev["dpos"]),
                delta_neg=float(ev["dneg"]),
                tag="cv_last_fold",
                source_csv=csv_src,
            )
        if do_plots and args.cv_plot_last and last_qcv is not None:
            ev = last_qcv["ev"]
            dpos_q, dneg_q = ev["dpos"], ev["dneg"]
            gs_tr, gs_te = last_qcv["gs_tr"], last_qcv["gs_te"]
            ld_tr, ld_te = last_qcv["ld_tr"], last_qcv["ld_te"]
            pred_tr, pred_te = last_qcv["pred_tr"], last_qcv["pred_te"]
            q_meas_all = np.concatenate([gs_tr, gs_te])
            q_rate_all = np.concatenate([pred_tr, pred_te])
            is_train = np.concatenate(
                [np.ones(len(gs_tr), dtype=bool), np.zeros(len(gs_te), dtype=bool)]
            )
            load_all = np.concatenate([ld_tr, ld_te])
            base = plot_dir / "qlattice_qcr_turner_labels_cv_last"
            plot_turner_qmeas_vs_qrate(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_q,
                dneg_q,
                r"QLattice $Q_{\mathrm{cr}}$ vs $Q_{\mathrm{meas}}$ (qcr_labels last CV fold)",
                r"Symbolic $Q_{\mathrm{cr}}$ (m³/day)",
                base,
                loading=load_all,
                show=show_plots,
            )
            plot_residual_qminus_qmeas(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_q,
                dneg_q,
                r"Residual: $Q_{\mathrm{cr, pred}} - Q_{\mathrm{meas}}$ (last CV fold)",
                base,
                loading=load_all,
                show=show_plots,
            )
        qcr_labels_ran_cv = True

    if run_qcr_custom:
        print(
            "\n========== QLattice mode B: symbolic Q_critical (teacher = optional .npz) =========="
        )
        print(
            "  True Q_critical is unknown. Feyn learns a closed-form estimate; with a .npz, training "
            "targets come from your teacher arrays. Loading evaluation uses (symbolic_pred - Q_meas) vs delta vs labels."
        )
        npz_path = args.qcr_npz
        if npz_path is None:
            npz_path = default_qcr_npz_path()
        if not npz_path.is_file():
            print(
                f"  Skipping: file not found: {npz_path}\n"
                "  For a self-contained run use --mode qcr_labels (no .npz).\n"
                "  For qcr_custom, pass --qcr-npz with y_train_pred / y_test_pred matching this split "
                "(see load_qcr_train_test_from_npz)."
            )
        else:
            _npz = Path(npz_path)
            _sz = _npz.stat().st_size
            if _sz == 0:
                raise SystemExit(
                    f"{_npz} is empty (0 bytes). Remove it, or use --mode qcr_labels, "
                    "or replace with a valid .npz (y_train_pred / y_test_pred, same split as this script)."
                )
            try:
                z = np.load(_npz, allow_pickle=False)
            except (EOFError, OSError, zipfile.BadZipFile, ValueError) as exc:
                raise SystemExit(
                    f"Cannot read {_npz} ({type(exc).__name__}: {exc}). "
                    "The file is corrupted or not a valid .npz. Delete it or fix --qcr-npz; "
                    "arrays must match RANDOM_STATE=42 split (y_train_pred / y_test_pred). "
                    "Or use --mode qcr_labels and omit the .npz."
                ) from exc
            ytr, yte = load_qcr_train_test_from_npz(z)
            if len(ytr) != len(data["gs_train"]) or len(yte) != len(data["gs_test"]):
                raise SystemExit(
                    "y_train_pred / y_test_pred length must match current train/test split. "
                    "Re-export npz using the same CSV and RANDOM_STATE=42 split, or run training first."
                )
            qcr_tr, qcr_te = append_measured_gas_feature(
                train_X_df,
                test_X_df,
                data["gs_train"],
                data["gs_test"],
                bool(getattr(args, "qcr_use_measured_gas", False)),
            )
            if args.qcr_use_measured_gas:
                print(
                    "  Using --qcr-use-measured-gas: Gasflowrate_meas is an input (same as Q_meas at that well)."
                )
            best_q, pred_tr_q, pred_te_q, _, _, _ = feyn_run_qcritical_fit(
                qcr_tr,
                qcr_te,
                data["gs_train"],
                data["load_train"],
                ytr,
                args,
                feyn,
                args.seed + 7,
            )
            mae_c = float(np.mean(np.abs(pred_te_q - yte)))
            print(f"  Test MAE (symbolic vs physics-model Q_critical targets): {mae_c:,.2f} m3/day")
            print(
                f"  mean|physics-model Q_critical - Q_meas| on test = "
                f"{float(np.mean(np.abs(yte - data['gs_test']))):,.2f}  (both are estimates; true Q_cr unknown)"
            )
            _print_model_eq(
                best_q,
                "Symbolic Q_critical ~ f(features) — Feyn fit to physics-model teacher (.npz)",
                note="Operational rule vs labels: (symbolic_pred - Q_meas) vs delta (same style as notebook).",
            )
            dpos_q, dneg_q = class_metrics_turner(
                pred_tr_q,
                data["gs_train"],
                data["load_train"],
                pred_te_q,
                data["gs_test"],
                data["load_test"],
                "Turner (symbolic Q_critical_pred vs Q_meas, delta bands)",
            )
            if export_csv:
                q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
                q_rate_all = np.concatenate([pred_tr_q, pred_te_q])
                is_train = np.concatenate(
                    [
                        np.ones(len(data["gs_train"]), dtype=bool),
                        np.zeros(len(data["gs_test"]), dtype=bool),
                    ]
                )
                load_all = np.concatenate([data["load_train"], data["load_test"]])
                save_pipeline_predictions_csv(
                    export_dir,
                    pipeline="qlattice",
                    mode="qcr_custom",
                    seed=args.seed,
                    q_meas=q_meas_all,
                    q_pred=q_rate_all,
                    loading=load_all,
                    is_train=is_train,
                    pred_column="Q_cr_pred_m3day",
                    delta_pos=float(dpos_q),
                    delta_neg=float(dneg_q),
                    source_csv=csv_src,
                )
            if do_plots:
                q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
                q_rate_all = np.concatenate([pred_tr_q, pred_te_q])
                is_train = np.concatenate(
                    [
                        np.ones(len(data["gs_train"]), dtype=bool),
                        np.zeros(len(data["gs_test"]), dtype=bool),
                    ]
                )
                load_all = np.concatenate([data["load_train"], data["load_test"]])
                base = plot_dir / "qlattice_qcr_turner"
                plot_turner_qmeas_vs_qrate(
                    q_meas_all,
                    q_rate_all,
                    is_train,
                    dpos_q,
                    dneg_q,
                    r"QLattice $Q_{\mathrm{cr}}$ vs $Q_{\mathrm{meas}}$ (Turner $\delta$ bands)",
                    r"Symbolic $Q_{\mathrm{cr}}$ (m³/day)",
                    base,
                    loading=load_all,
                    show=show_plots,
                )
                plot_residual_qminus_qmeas(
                    q_meas_all,
                    q_rate_all,
                    is_train,
                    dpos_q,
                    dneg_q,
                    r"Residual: $Q_{\mathrm{cr, pred}} - Q_{\mathrm{meas}}$ (Turner margin)",
                    base,
                    loading=load_all,
                    show=show_plots,
                )

    if run_qcr_labels and not qcr_labels_ran_cv:
        pscale = float(getattr(args, "qcr_proxy_margin_scale", 1.0))
        if pscale <= 0:
            raise SystemExit("--qcr-proxy-margin-scale must be positive.")
        print(
            "\n========== QLattice: symbolic Q_critical (label-implied targets, no .npz) =========="
        )
        print(
            "  Train target per well: Q_meas + LOSS_DELTA for loaded, Q_meas - LOSS_DELTA for unloaded, "
            "else Q_meas (near). This does **not** use your other code's Q_cr or any physics-model export."
        )
        print(
            "  On the Turner cross-plot, ideal loaded/unloaded sit only ~±δ above/below the 1:1 line — δ is "
            "often a modest **fraction** of high Q_meas, so points look 'close' to the diagonal even when the "
            "rule is working. Near-L.U points are **supposed** to sit in the yellow strip. "
            "Check the residual plot and the class-wise means below."
        )
        if pscale != 1.0:
            print(f"  Proxy margin scale = {pscale} (loaded/unloaded offsets × this; Turner δ unchanged).")
        print(
            "  Accuracy tips: --cv-folds 5 for K-fold Turner scores; --select-by hybrid or mae; "
            "raise --epochs / --max-complexity / --top-models; --loss squared_error sometimes helps. "
            "Optional --qcr-use-measured-gas adds Q_meas as an input (strong fit, not geometry-only). "
            "Macro-F1 stays low if the Near class is rare or never predicted (narrow strip vs model spread)."
        )
        qcr_tr, qcr_te = append_measured_gas_feature(
            train_X_df,
            test_X_df,
            data["gs_train"],
            data["gs_test"],
            bool(getattr(args, "qcr_use_measured_gas", False)),
        )
        if args.qcr_use_measured_gas:
            print(
                "  Using --qcr-use-measured-gas: Gasflowrate_meas is an input (same as Q_meas at that well)."
            )
        ytr_lab = q_critical_label_proxy_targets(
            data["gs_train"], data["load_train"], margin_scale=pscale
        )
        yte_lab = q_critical_label_proxy_targets(data["gs_test"], data["load_test"], margin_scale=pscale)
        best_q, pred_tr_q, pred_te_q, _, _, _ = feyn_run_qcritical_fit(
            qcr_tr,
            qcr_te,
            data["gs_train"],
            data["load_train"],
            ytr_lab,
            args,
            feyn,
            args.seed + 7,
        )
        mae_lab = float(np.mean(np.abs(pred_te_q - yte_lab)))
        print(
            f"  Test MAE (symbolic vs label-proxy target on held-out wells): {mae_lab:,.2f} m3/day  "
            "(proxy uses true test labels, not predictions — for calibration only)"
        )
        print_mean_residual_by_loading(pred_te_q, data["gs_test"], data["load_test"], "test")
        _print_model_eq(
            best_q,
            "Symbolic Q_critical ~ f(features) — Feyn fit to Q_meas +/- margin by loading class",
            note="Operational rule vs labels: (symbolic_pred - Q_meas) vs delta.",
        )
        dpos_q, dneg_q = class_metrics_turner(
            pred_tr_q,
            data["gs_train"],
            data["load_train"],
            pred_te_q,
            data["gs_test"],
            data["load_test"],
            "Turner (symbolic Q_critical_pred vs Q_meas, delta bands)",
        )
        if export_csv:
            q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
            q_rate_all = np.concatenate([pred_tr_q, pred_te_q])
            is_train = np.concatenate(
                [
                    np.ones(len(data["gs_train"]), dtype=bool),
                    np.zeros(len(data["gs_test"]), dtype=bool),
                ]
            )
            load_all = np.concatenate([data["load_train"], data["load_test"]])
            save_pipeline_predictions_csv(
                export_dir,
                pipeline="qlattice",
                mode="qcr_labels",
                seed=args.seed,
                q_meas=q_meas_all,
                q_pred=q_rate_all,
                loading=load_all,
                is_train=is_train,
                pred_column="Q_cr_pred_m3day",
                delta_pos=float(dpos_q),
                delta_neg=float(dneg_q),
                source_csv=csv_src,
            )
        if do_plots:
            q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
            q_rate_all = np.concatenate([pred_tr_q, pred_te_q])
            is_train = np.concatenate(
                [
                    np.ones(len(data["gs_train"]), dtype=bool),
                    np.zeros(len(data["gs_test"]), dtype=bool),
                ]
            )
            load_all = np.concatenate([data["load_train"], data["load_test"]])
            base = plot_dir / "qlattice_qcr_turner_labels"
            plot_turner_qmeas_vs_qrate(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_q,
                dneg_q,
                r"QLattice $Q_{\mathrm{cr}}$ vs $Q_{\mathrm{meas}}$ (label-proxy fit, Turner $\delta$)",
                r"Symbolic $Q_{\mathrm{cr}}$ (m³/day)",
                base,
                loading=load_all,
                show=show_plots,
            )
            plot_residual_qminus_qmeas(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_q,
                dneg_q,
                r"Residual: $Q_{\mathrm{cr, pred}} - Q_{\mathrm{meas}}$ (Turner margin)",
                base,
                loading=load_all,
                show=show_plots,
            )

    print("\nDone.")

if __name__ == "__main__":
    main()