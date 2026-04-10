#!/usr/bin/env python3
"""
TabPFN regression on the same wells / split / Turner metrics as ``qlattice_pipeline.py``.

**``qcr_labels``** builds label-proxy Q_critical targets inside this module (no hand-off from other
notebooks). **``qcr_custom``** is optional if you supply a matching ``.npz`` teacher yourself.

Install: pip install tabpfn pandas numpy scikit-learn matplotlib torch

Example:
  python train_eval/tabpfn_standalone.py --mode gas --plot-dir .
  python train_eval/tabpfn_standalone.py --mode qcr_labels --qcr-proxy-margin-scale 2.5 --plot-dir .
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .qlattice_pipeline import (
    append_measured_gas_feature,
    class_metrics_turner,
    default_processed_well_csv,
    default_qcr_npz_path,
    eval_turner_train_test_metrics,
    fmt_mean_std,
    fold_train_test_frames,
    load_and_split,
    load_filtered_unscaled,
    load_qcr_train_test_from_npz,
    make_cv_folds,
    plot_residual_qminus_qmeas,
    plot_turner_qmeas_vs_qrate,
    print_mean_residual_by_loading,
    q_critical_label_proxy_targets,
    save_pipeline_predictions_csv,
    set_loss_margins,
)

try:
    from tabpfn import TabPFNRegressor
except ImportError:  # pragma: no cover
    TabPFNRegressor = None  # type: ignore[misc, assignment]


def _ensure_tabpfn() -> type:
    if TabPFNRegressor is None:
        raise SystemExit(
            "Missing package: pip install tabpfn\nSee https://pypi.org/project/tabpfn/"
        )
    return TabPFNRegressor


def _gas_target_forward(gs: np.ndarray, log1p_gas: bool) -> np.ndarray:
    g = np.asarray(gs, dtype=float)
    if log1p_gas:
        return np.log1p(np.maximum(g, 0.0))
    return g


def _gas_pred_inverse(pred: np.ndarray, log1p_gas: bool) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    if log1p_gas:
        return np.maximum(0.0, np.expm1(np.clip(p, -50.0, 50.0)))
    return p


def frames_to_X(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(train_df.values, dtype=np.float64)
    X_test = np.asarray(test_df.values, dtype=np.float64)
    return X_train, X_test


def fit_predict_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    reg: type,
    random_state: int,
    ignore_limits: bool,
    n_estimators: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_tr = np.asarray(y_train, dtype=np.float64).ravel()
    model = reg(
        n_estimators=n_estimators,
        device=device,
        random_state=random_state,
        ignore_pretraining_limits=ignore_limits,
    )
    model.fit(X_train, y_tr)
    pred_tr = np.asarray(model.predict(X_train), dtype=float).ravel()
    pred_te = np.asarray(model.predict(X_test), dtype=float).ravel()
    return pred_tr, pred_te


def train_predict_gas_tabpfn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gs_train: np.ndarray,
    gs_test: np.ndarray,
    load_train: np.ndarray,
    load_test: np.ndarray,
    args: argparse.Namespace,
    Regressor: type,
) -> dict:
    log1p_gas = not bool(getattr(args, "linear_gas", False))
    X_train, X_test = frames_to_X(train_df, test_df)
    y_train = _gas_target_forward(gs_train, log1p_gas)
    pred_tr, pred_te = fit_predict_tabpfn(
        X_train,
        y_train,
        X_test,
        reg=Regressor,
        random_state=int(args.seed),
        ignore_limits=bool(args.tabpfn_ignore_limits),
        n_estimators=int(args.tabpfn_n_estimators),
        device=str(args.device),
    )
    pred_tr = _gas_pred_inverse(pred_tr, log1p_gas)
    pred_te = _gas_pred_inverse(pred_te, log1p_gas)
    mae = float(np.mean(np.abs(pred_te - gs_test)))
    rmse = float(np.sqrt(np.mean((pred_te - gs_test) ** 2)))
    ev = eval_turner_train_test_metrics(pred_tr, gs_train, load_train, pred_te, gs_test, load_test)
    return {
        "mae": mae,
        "rmse": rmse,
        "pred_tr": pred_tr,
        "pred_te": pred_te,
        **ev,
    }


def train_predict_qcr_tabpfn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gs_train: np.ndarray,
    gs_test: np.ndarray,
    load_train: np.ndarray,
    load_test: np.ndarray,
    y_train_target: np.ndarray,
    args: argparse.Namespace,
    Regressor: type,
    *,
    fit_random_state: int | None = None,
) -> dict:
    X_train, X_test = frames_to_X(train_df, test_df)
    y_train = np.asarray(y_train_target, dtype=np.float64).ravel()
    rs = int(args.seed) + 3 if fit_random_state is None else int(fit_random_state)
    pred_tr, pred_te = fit_predict_tabpfn(
        X_train,
        y_train,
        X_test,
        reg=Regressor,
        random_state=rs,
        ignore_limits=bool(args.tabpfn_ignore_limits),
        n_estimators=int(args.tabpfn_n_estimators),
        device=str(args.device),
    )
    ev = eval_turner_train_test_metrics(pred_tr, gs_train, load_train, pred_te, gs_test, load_test)
    return {
        "pred_tr": pred_tr,
        "pred_te": pred_te,
        **ev,
    }


def build_feature_frames_from_split(
    data: dict,
    *,
    physical_units: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fc = data["feature_cols"]
    if physical_units:
        X_train = data["scaler"].inverse_transform(data["X_train"])
        X_test = data["scaler"].inverse_transform(data["X_test"])
    else:
        X_train = data["X_train"]
        X_test = data["X_test"]
    return (
        pd.DataFrame(X_train, columns=fc),
        pd.DataFrame(X_test, columns=fc),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TabPFN regression: same split and Turner metrics as qlattice_pipeline.py"
    )
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
        help="gas = measured rate. qcr_custom = .npz teacher. qcr_labels = label-proxy. both = gas + qcr_custom (needs .npz).",
    )
    p.add_argument("--qcr-npz", type=Path, default=None, help="Physics-model Q_critical .npz (qcr_custom / both).")
    p.add_argument(
        "--qcr-proxy-margin-scale",
        type=float,
        default=1.0,
        help="qcr_labels: scale loaded/unloaded proxy offsets.",
    )
    p.add_argument(
        "--qcr-use-measured-gas",
        action="store_true",
        help="Append Gasflowrate_meas as a column for Q_critical modes.",
    )
    p.add_argument(
        "--features",
        choices=("physical", "scaled"),
        default="physical",
        help="physical = inverse StandardScaler (same as Feyn). scaled = scaled X from split.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--linear-gas",
        action="store_true",
        help="Regress raw Gasflowrate; default log1p then expm1 for metrics.",
    )
    p.add_argument("--tabpfn-n-estimators", type=int, default=8, help="TabPFN ensemble size.")
    p.add_argument("--device", type=str, default="auto", help="TabPFN device: auto, cpu, cuda, ...")
    p.add_argument(
        "--tabpfn-ignore-limits",
        action="store_true",
        help="ignore_pretraining_limits=True if fit fails on sample limits.",
    )
    p.add_argument("--cv-folds", type=int, default=0, help="If >=2, k-fold CV for gas and/or qcr_labels.")
    p.add_argument("--cv-plot-last", action="store_true", help="Save Turner plots for last CV fold only.")
    p.add_argument("--plot-dir", type=Path, default=None, help="Plot output directory (default: cwd).")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--show-plots", action="store_true")
    p.add_argument(
        "--export-csv-dir",
        type=Path,
        default=None,
        help="Write Q_meas + prediction columns to this directory (default: same as --plot-dir).",
    )
    p.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Do not write prediction CSV files.",
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

    Regressor = _ensure_tabpfn()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    data = load_and_split(args.csv, args.seed)
    physical = args.features == "physical"
    train_df, test_df = build_feature_frames_from_split(data, physical_units=physical)

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

    raw_for_cv = None
    if run_gas and cv_k >= 2:
        raw_for_cv = load_filtered_unscaled(args.csv)

    if run_gas and cv_k >= 2 and raw_for_cv is not None:
        print("\n========== TabPFN gas — K-fold cross-validation ==========")
        kind, folds, k_eff = make_cv_folds(raw_for_cv["stratify"], cv_k, args.seed)
        print(f"  {k_eff} folds ({kind}), N={len(raw_for_cv['gsflow'])} wells.")
        maes, rmses, accs_tr, accs_te, f1ms, f1lus = [], [], [], [], [], []
        last = None
        for fi, (tr_i, te_i) in enumerate(folds):
            med_fold = float(np.nanmedian(raw_for_cv["gsflow"][tr_i]))
            set_loss_margins(med_fold)
            tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te = fold_train_test_frames(raw_for_cv, tr_i, te_i)
            out = train_predict_gas_tabpfn(tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te, args, Regressor)
            maes.append(out["mae"])
            rmses.append(out["rmse"])
            accs_tr.append(out["acc_train"])
            accs_te.append(out["acc"])
            f1ms.append(out["f1_macro"])
            f1lus.append(out["f1_lu"])
            print(
                f"  Fold {fi + 1}/{k_eff}: train Acc={out['acc_train']:.4f}  test Acc={out['acc']:.4f}  "
                f"MAE={out['mae']:,.2f}  macro-F1={out['f1_macro']:.4f}  mean-F1(L,U)={out['f1_lu']:.4f}"
            )
            last = (tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te, out)
        print("\n  --- CV summary (gas) ---")
        print(f"  MAE (m³/day):     {fmt_mean_std(maes)}")
        print(f"  RMSE:             {fmt_mean_std(rmses)}")
        print(f"  Train Acc:        {fmt_mean_std(accs_tr)}")
        print(f"  Test Acc:         {fmt_mean_std(accs_te)}")
        print(f"  macro-F1 (test):  {fmt_mean_std(f1ms)}")
        print(f"  mean-F1(L,U):     {fmt_mean_std(f1lus)}")
        if export_csv and last is not None:
            tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te, out = last
            save_pipeline_predictions_csv(
                export_dir,
                pipeline="tabpfn",
                mode="gas",
                seed=args.seed,
                q_meas=np.concatenate([gs_tr, gs_te]),
                q_pred=np.concatenate([out["pred_tr"], out["pred_te"]]),
                loading=np.concatenate([ld_tr, ld_te]),
                is_train=np.concatenate(
                    [np.ones(len(gs_tr), dtype=bool), np.zeros(len(gs_te), dtype=bool)]
                ),
                pred_column="Q_gas_pred_m3day",
                delta_pos=float(out["dpos"]),
                delta_neg=float(out["dneg"]),
                tag="cv_last_fold",
                source_csv=csv_src,
            )
        if do_plots and args.cv_plot_last and last is not None:
            tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te, out = last
            dpos_g, dneg_g = out["dpos"], out["dneg"]
            q_meas_all = np.concatenate([gs_tr, gs_te])
            q_rate_all = np.concatenate([out["pred_tr"], out["pred_te"]])
            is_train = np.concatenate([np.ones(len(gs_tr), dtype=bool), np.zeros(len(gs_te), dtype=bool)])
            load_all = np.concatenate([ld_tr, ld_te])
            base = plot_dir / "tabpfn_gas_turner_cv_last"
            plot_turner_qmeas_vs_qrate(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_g,
                dneg_g,
                "TabPFN gas vs measured — last CV fold",
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
                "Residual: TabPFN gas - Q_meas (last CV fold)",
                base,
                loading=load_all,
                show=show_plots,
            )
    elif run_gas:
        print("\n========== TabPFN gas (measured rate) ==========")
        print(f"  Features: {args.features}  |  log1p target: {not args.linear_gas}")
        out_g = train_predict_gas_tabpfn(
            train_df,
            test_df,
            data["gs_train"],
            data["gs_test"],
            data["load_train"],
            data["load_test"],
            args,
            Regressor,
        )
        print(f"  Test MAE: {out_g['mae']:,.2f} m³/day  |  RMSE: {out_g['rmse']:,.2f}")
        class_metrics_turner(
            out_g["pred_tr"],
            data["gs_train"],
            data["load_train"],
            out_g["pred_te"],
            data["gs_test"],
            data["load_test"],
            "Turner (TabPFN gas vs Q_meas)",
        )
        if export_csv:
            q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
            q_rate_all = np.concatenate([out_g["pred_tr"], out_g["pred_te"]])
            is_train = np.concatenate(
                [np.ones(len(data["gs_train"]), dtype=bool), np.zeros(len(data["gs_test"]), dtype=bool)]
            )
            load_all = np.concatenate([data["load_train"], data["load_test"]])
            save_pipeline_predictions_csv(
                export_dir,
                pipeline="tabpfn",
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
                [np.ones(len(data["gs_train"]), dtype=bool), np.zeros(len(data["gs_test"]), dtype=bool)]
            )
            load_all = np.concatenate([data["load_train"], data["load_test"]])
            base = plot_dir / "tabpfn_gas_turner"
            dpos_g, dneg_g = out_g["dpos"], out_g["dneg"]
            plot_turner_qmeas_vs_qrate(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_g,
                dneg_g,
                "TabPFN gas vs measured (Turner margins)",
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
                "Residual: TabPFN gas - Q_meas",
                base,
                loading=load_all,
                show=show_plots,
            )

    if run_qcr_labels and cv_k >= 2:
        print("\n========== TabPFN qcr_labels — K-fold cross-validation ==========")
        raw_q = load_filtered_unscaled(args.csv)
        kind, folds, k_eff = make_cv_folds(raw_q["stratify"], cv_k, args.seed)
        print(f"  {k_eff} folds ({kind}), N={len(raw_q['gsflow'])} wells.")
        pscale = float(args.qcr_proxy_margin_scale)
        if pscale <= 0:
            raise SystemExit("--qcr-proxy-margin-scale must be positive.")
        accs_tr, accs_te, f1ms, f1lus, maes = [], [], [], [], []
        last_q = None
        for fi, (tr_i, te_i) in enumerate(folds):
            med_fold = float(np.nanmedian(raw_q["gsflow"][tr_i]))
            set_loss_margins(med_fold)
            tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te = fold_train_test_frames(raw_q, tr_i, te_i)
            tr_df, te_df = append_measured_gas_feature(tr_df, te_df, gs_tr, gs_te, bool(args.qcr_use_measured_gas))
            ytr = q_critical_label_proxy_targets(gs_tr, ld_tr, margin_scale=pscale)
            yte_ref = q_critical_label_proxy_targets(gs_te, ld_te, margin_scale=pscale)
            out = train_predict_qcr_tabpfn(
                tr_df,
                te_df,
                gs_tr,
                gs_te,
                ld_tr,
                ld_te,
                ytr,
                args,
                Regressor,
                fit_random_state=args.seed + 7 + fi * 97,
            )
            mae_te = float(np.mean(np.abs(out["pred_te"] - yte_ref)))
            maes.append(mae_te)
            accs_tr.append(out["acc_train"])
            accs_te.append(out["acc"])
            f1ms.append(out["f1_macro"])
            f1lus.append(out["f1_lu"])
            print(
                f"  Fold {fi + 1}/{k_eff}: train Acc={out['acc_train']:.4f}  test Acc={out['acc']:.4f}  "
                f"MAE vs proxy={mae_te:,.2f}  macro-F1={out['f1_macro']:.4f}"
            )
            last_q = (tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te, out)
        print("\n  --- CV summary (qcr_labels) ---")
        print(f"  MAE vs proxy (test): {fmt_mean_std(maes)}")
        print(f"  Train Acc:           {fmt_mean_std(accs_tr)}")
        print(f"  Test Acc:            {fmt_mean_std(accs_te)}")
        print(f"  macro-F1 (test):     {fmt_mean_std(f1ms)}")
        print(f"  mean-F1(L,U):        {fmt_mean_std(f1lus)}")
        if export_csv and last_q is not None:
            tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te, out = last_q
            save_pipeline_predictions_csv(
                export_dir,
                pipeline="tabpfn",
                mode="qcr_labels",
                seed=args.seed,
                q_meas=np.concatenate([gs_tr, gs_te]),
                q_pred=np.concatenate([out["pred_tr"], out["pred_te"]]),
                loading=np.concatenate([ld_tr, ld_te]),
                is_train=np.concatenate(
                    [np.ones(len(gs_tr), dtype=bool), np.zeros(len(gs_te), dtype=bool)]
                ),
                pred_column="Q_cr_pred_m3day",
                delta_pos=float(out["dpos"]),
                delta_neg=float(out["dneg"]),
                tag="cv_last_fold",
                source_csv=csv_src,
            )
        if do_plots and args.cv_plot_last and last_q is not None:
            tr_df, te_df, gs_tr, gs_te, ld_tr, ld_te, out = last_q
            dpos_q, dneg_q = out["dpos"], out["dneg"]
            q_meas_all = np.concatenate([gs_tr, gs_te])
            q_rate_all = np.concatenate([out["pred_tr"], out["pred_te"]])
            is_train = np.concatenate([np.ones(len(gs_tr), dtype=bool), np.zeros(len(gs_te), dtype=bool)])
            load_all = np.concatenate([ld_tr, ld_te])
            base = plot_dir / "tabpfn_qcr_labels_turner_cv_last"
            plot_turner_qmeas_vs_qrate(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_q,
                dneg_q,
                r"TabPFN $Q_{\mathrm{cr}}$ vs $Q_{\mathrm{meas}}$ (last CV fold)",
                r"Predicted $Q_{\mathrm{cr}}$ (m³/day)",
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
                r"Residual: $Q_{\mathrm{cr}}$ - $Q_{\mathrm{meas}}$ (last CV fold)",
                base,
                loading=load_all,
                show=show_plots,
            )
        qcr_labels_ran_cv = True

    if run_qcr_custom:
        print("\n========== TabPFN qcr_custom (physics teacher .npz) ==========")
        npz_path = args.qcr_npz if args.qcr_npz is not None else default_qcr_npz_path()
        if not npz_path.is_file():
            print(f"  Skipping: file not found: {npz_path}")
        else:
            _npz = Path(npz_path)
            if _npz.stat().st_size == 0:
                raise SystemExit(f"{_npz} is empty (0 bytes).")
            try:
                z = np.load(_npz, allow_pickle=False)
            except (EOFError, OSError, zipfile.BadZipFile, ValueError) as exc:
                raise SystemExit(f"Cannot read {_npz}: {exc}") from exc
            ytr, yte = load_qcr_train_test_from_npz(z)
            if len(ytr) != len(data["gs_train"]) or len(yte) != len(data["gs_test"]):
                raise SystemExit("npz row counts must match current train/test split.")
            qcr_tr, qcr_te = append_measured_gas_feature(
                train_df,
                test_df,
                data["gs_train"],
                data["gs_test"],
                bool(args.qcr_use_measured_gas),
            )
            out = train_predict_qcr_tabpfn(
                qcr_tr,
                qcr_te,
                data["gs_train"],
                data["gs_test"],
                data["load_train"],
                data["load_test"],
                ytr,
                args,
                Regressor,
            )
            mae_c = float(np.mean(np.abs(out["pred_te"] - yte)))
            print(f"  Test MAE (TabPFN vs physics-model Q_cr): {mae_c:,.2f} m³/day")
            class_metrics_turner(
                out["pred_tr"],
                data["gs_train"],
                data["load_train"],
                out["pred_te"],
                data["gs_test"],
                data["load_test"],
                "Turner (TabPFN Q_cr vs Q_meas)",
            )
            if export_csv:
                q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
                q_rate_all = np.concatenate([out["pred_tr"], out["pred_te"]])
                is_train = np.concatenate(
                    [np.ones(len(data["gs_train"]), dtype=bool), np.zeros(len(data["gs_test"]), dtype=bool)]
                )
                load_all = np.concatenate([data["load_train"], data["load_test"]])
                save_pipeline_predictions_csv(
                    export_dir,
                    pipeline="tabpfn",
                    mode="qcr_custom",
                    seed=args.seed,
                    q_meas=q_meas_all,
                    q_pred=q_rate_all,
                    loading=load_all,
                    is_train=is_train,
                    pred_column="Q_cr_pred_m3day",
                    delta_pos=float(out["dpos"]),
                    delta_neg=float(out["dneg"]),
                    source_csv=csv_src,
                )
            if do_plots:
                q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
                q_rate_all = np.concatenate([out["pred_tr"], out["pred_te"]])
                is_train = np.concatenate(
                    [np.ones(len(data["gs_train"]), dtype=bool), np.zeros(len(data["gs_test"]), dtype=bool)]
                )
                load_all = np.concatenate([data["load_train"], data["load_test"]])
                base = plot_dir / "tabpfn_qcr_custom_turner"
                dpos_q, dneg_q = out["dpos"], out["dneg"]
                plot_turner_qmeas_vs_qrate(
                    q_meas_all,
                    q_rate_all,
                    is_train,
                    dpos_q,
                    dneg_q,
                    r"TabPFN $Q_{\mathrm{cr}}$ vs $Q_{\mathrm{meas}}$ (teacher .npz)",
                    r"Predicted $Q_{\mathrm{cr}}$ (m³/day)",
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
                    r"Residual: $Q_{\mathrm{cr}}$ - $Q_{\mathrm{meas}}$",
                    base,
                    loading=load_all,
                    show=show_plots,
                )

    if run_qcr_labels and not qcr_labels_ran_cv:
        print("\n========== TabPFN qcr_labels (label-proxy targets) ==========")
        pscale = float(args.qcr_proxy_margin_scale)
        if pscale <= 0:
            raise SystemExit("--qcr-proxy-margin-scale must be positive.")
        qcr_tr, qcr_te = append_measured_gas_feature(
            train_df,
            test_df,
            data["gs_train"],
            data["gs_test"],
            bool(args.qcr_use_measured_gas),
        )
        ytr_lab = q_critical_label_proxy_targets(data["gs_train"], data["load_train"], margin_scale=pscale)
        yte_lab = q_critical_label_proxy_targets(data["gs_test"], data["load_test"], margin_scale=pscale)
        out = train_predict_qcr_tabpfn(
            qcr_tr,
            qcr_te,
            data["gs_train"],
            data["gs_test"],
            data["load_train"],
            data["load_test"],
            ytr_lab,
            args,
            Regressor,
        )
        mae_lab = float(np.mean(np.abs(out["pred_te"] - yte_lab)))
        print(f"  Test MAE (vs label-proxy target): {mae_lab:,.2f} m³/day")
        print_mean_residual_by_loading(out["pred_te"], data["gs_test"], data["load_test"], "test")
        class_metrics_turner(
            out["pred_tr"],
            data["gs_train"],
            data["load_train"],
            out["pred_te"],
            data["gs_test"],
            data["load_test"],
            "Turner (TabPFN Q_cr vs Q_meas)",
        )
        if export_csv:
            q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
            q_rate_all = np.concatenate([out["pred_tr"], out["pred_te"]])
            is_train = np.concatenate(
                [np.ones(len(data["gs_train"]), dtype=bool), np.zeros(len(data["gs_test"]), dtype=bool)]
            )
            load_all = np.concatenate([data["load_train"], data["load_test"]])
            save_pipeline_predictions_csv(
                export_dir,
                pipeline="tabpfn",
                mode="qcr_labels",
                seed=args.seed,
                q_meas=q_meas_all,
                q_pred=q_rate_all,
                loading=load_all,
                is_train=is_train,
                pred_column="Q_cr_pred_m3day",
                delta_pos=float(out["dpos"]),
                delta_neg=float(out["dneg"]),
                source_csv=csv_src,
            )
        if do_plots:
            q_meas_all = np.concatenate([data["gs_train"], data["gs_test"]])
            q_rate_all = np.concatenate([out["pred_tr"], out["pred_te"]])
            is_train = np.concatenate(
                [np.ones(len(data["gs_train"]), dtype=bool), np.zeros(len(data["gs_test"]), dtype=bool)]
            )
            load_all = np.concatenate([data["load_train"], data["load_test"]])
            base = plot_dir / "tabpfn_qcr_labels_turner"
            dpos_q, dneg_q = out["dpos"], out["dneg"]
            plot_turner_qmeas_vs_qrate(
                q_meas_all,
                q_rate_all,
                is_train,
                dpos_q,
                dneg_q,
                r"TabPFN $Q_{\mathrm{cr}}$ vs $Q_{\mathrm{meas}}$ (label-proxy)",
                r"Predicted $Q_{\mathrm{cr}}$ (m³/day)",
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
                r"Residual: $Q_{\mathrm{cr}}$ - $Q_{\mathrm{meas}}$",
                base,
                loading=load_all,
                show=show_plots,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
