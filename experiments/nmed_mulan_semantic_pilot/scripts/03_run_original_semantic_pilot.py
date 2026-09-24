from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    ExperimentConfig,
    align_targets_and_features,
    build_original_group_timeseries,
    numeric_feature_columns,
    read_nmed_window_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run original-only semantic/acoustic encoding pilot.")
    parser.add_argument("--window-csv", default="data/external/nmed_window_level_data.csv")
    parser.add_argument("--feature-csv", default=None)
    parser.add_argument("--feature-kind", default="features", help="Label used in output filenames.")
    parser.add_argument("--target", default="engagement_fast60")
    parser.add_argument("--pca-components", type=int, default=0, help="Use PCA before ridge when > 0.")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--figure-dir", default="outputs/figures")
    return parser.parse_args()


def make_time_series_cv(n_rows: int, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(n_splits, max(2, n_rows // 12))
    test_size = n_rows // (n_splits + 1)
    splits = []
    for fold in range(n_splits):
        test_start = n_rows - (n_splits - fold) * test_size
        test_end = test_start + test_size
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) >= 12 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    if not splits:
        raise ValueError(f"Not enough rows for time-series validation: {n_rows}")
    return splits


def cross_validated_predictions(x: np.ndarray, y: np.ndarray, pca_components: int) -> tuple[np.ndarray, list[dict]]:
    predictions = np.full(shape=len(y), fill_value=np.nan, dtype=float)
    fold_rows = []
    alphas = np.logspace(-3, 3, 13)
    for fold_id, (train_idx, test_idx) in enumerate(make_time_series_cv(len(y)), start=1):
        steps = [("scale", StandardScaler())]
        if pca_components > 0:
            n_components = min(pca_components, x.shape[1], len(train_idx) - 1)
            steps.append(("pca", PCA(n_components=n_components)))
        steps.append(("ridge", RidgeCV(alphas=alphas)))
        model = Pipeline(steps)
        model.fit(x[train_idx], y[train_idx])
        fold_pred = model.predict(x[test_idx])
        predictions[test_idx] = fold_pred
        fold_rows.append(
            {
                "fold": fold_id,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "test_start_window": int(test_idx[0]),
                "test_end_window": int(test_idx[-1]),
            }
        )
    return predictions, fold_rows


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return {
        "n_predicted": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic),
        "pearson_p": float(pearsonr(y_true, y_pred).pvalue),
        "spearman_rho": float(spearmanr(y_true, y_pred).statistic),
        "spearman_p": float(spearmanr(y_true, y_pred).pvalue),
    }


def save_prediction_plot(predictions: pd.DataFrame, target: str, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(predictions["center_sec"], predictions[target], label=f"Observed {target}", linewidth=2)
    ax.plot(predictions["center_sec"], predictions["prediction"], label="Cross-validated prediction", linewidth=2)
    ax.set_xlabel("Time in intact Elgar stimulus (s)")
    ax.set_ylabel(target)
    ax.legend(frameon=False)
    ax.set_title("NMED-E original-only semantic/acoustic pilot")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    window_df = read_nmed_window_table(args.window_csv)
    targets = build_original_group_timeseries(window_df, config=config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_csv = output_dir / "nmed_original_group_timeseries.csv"
    targets.to_csv(target_csv, index=False)

    if args.feature_csv is None:
        print(f"Wrote target time series only: {target_csv}")
        return

    feature_csv = Path(args.feature_csv)
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")
    features = pd.read_csv(feature_csv)
    feature_columns = numeric_feature_columns(features)
    aligned, feature_columns = align_targets_and_features(targets, features, args.target, feature_columns)
    x = aligned[feature_columns].to_numpy(dtype=float)
    y = aligned[args.target].to_numpy(dtype=float)
    predictions, fold_rows = cross_validated_predictions(x, y, pca_components=args.pca_components)
    aligned["prediction"] = predictions

    stem = f"{args.feature_kind}_{args.target}"
    predictions_csv = output_dir / f"{stem}_predictions.csv"
    metrics_json = output_dir / f"{stem}_metrics.json"
    folds_csv = output_dir / f"{stem}_cv_folds.csv"
    figure_path = Path(args.figure_dir) / f"{stem}_prediction.png"

    aligned.to_csv(predictions_csv, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_csv, index=False)
    metrics = {
        "target": args.target,
        "feature_kind": args.feature_kind,
        "feature_csv": str(feature_csv),
        "n_rows_after_alignment": int(len(aligned)),
        "n_features": int(len(feature_columns)),
        "pca_components": int(args.pca_components),
        **metric_dict(aligned[args.target].to_numpy(dtype=float), aligned["prediction"].to_numpy(dtype=float)),
    }
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_prediction_plot(aligned, args.target, figure_path)

    print(f"Wrote {predictions_csv}")
    print(f"Wrote {folds_csv}")
    print(f"Wrote {metrics_json}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
