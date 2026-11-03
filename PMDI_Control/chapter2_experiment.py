from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import imblearn
import xgboost
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_SPLITS = 5
AI4I_ZIP = "https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
SCANIA_ZIP = "https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip"

AI4I_TARGET = "Machine failure"
AI4I_ID_COLS = ["UDI", "Product ID"]
AI4I_FAILURE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
AI4I_TYPE = "Type"
AI4I_NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TREATMENTS = ["baseline", "weighting", "ros", "smote"]
MODELS = ["logreg", "rf", "xgb"]


@dataclass
class DatasetBundle:
    name: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    categorical: List[str]
    numeric: List[str]


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def download_and_extract(url: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "_download.zip"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    zip_path.unlink(missing_ok=True)


def ensure_data(data_dir: Path, download: bool) -> None:
    ai4i = data_dir / "ai4i2020.csv"
    sc_train = data_dir / "aps_failure_training_set.csv"
    sc_test = data_dir / "aps_failure_test_set.csv"

    if ai4i.exists() and sc_train.exists() and sc_test.exists():
        return
    if not download:
        missing = [str(p.name) for p in (ai4i, sc_train, sc_test) if not p.exists()]
        raise FileNotFoundError(
            "Missing data files: " + ", ".join(missing) +
            ". Put them in --data-dir or rerun with --download."
        )

    if not ai4i.exists():
        download_and_extract(AI4I_ZIP, data_dir)
    if not sc_train.exists() or not sc_test.exists():
        download_and_extract(SCANIA_ZIP, data_dir)


def read_scania_csv(path: Path) -> pd.DataFrame:
    """Read original UCI Scania CSV even if descriptive preamble is present."""
    header_idx = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("class,"):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"Could not locate Scania header in {path}")
    return pd.read_csv(path, skiprows=header_idx, na_values="na", low_memory=False)


def load_ai4i(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    required = set(AI4I_NUMERIC + [AI4I_TYPE, AI4I_TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"AI4I is missing columns: {sorted(missing)}")
    # Target-derived failure flags and identifiers are excluded to avoid leakage.
    drop_cols = [c for c in AI4I_ID_COLS + AI4I_FAILURE_FLAGS + [AI4I_TARGET] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    X = X[[AI4I_TYPE] + AI4I_NUMERIC]
    y = df[AI4I_TARGET].astype(int).copy()
    return X, y


def load_scania(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = read_scania_csv(path)
    if "class" not in df.columns:
        raise ValueError("Scania dataset does not contain 'class'.")
    y = df["class"].map({"neg": 0, "pos": 1})
    if y.isna().any():
        raise ValueError("Unexpected target values in Scania 'class'.")
    X = df.drop(columns=["class"]).apply(pd.to_numeric, errors="coerce")
    return X, y.astype(int)


def build_bundles(data_dir: Path) -> Dict[str, DatasetBundle]:
    X_ai, y_ai = load_ai4i(data_dir / "ai4i2020.csv")
    X_ai_train, X_ai_test, y_ai_train, y_ai_test = train_test_split(
        X_ai,
        y_ai,
        test_size=0.20,
        stratify=y_ai,
        random_state=RANDOM_STATE,
    )

    X_sc_train, y_sc_train = load_scania(data_dir / "aps_failure_training_set.csv")
    X_sc_test, y_sc_test = load_scania(data_dir / "aps_failure_test_set.csv")

    return {
        "AI4I": DatasetBundle(
            "AI4I",
            X_ai_train.reset_index(drop=True),
            y_ai_train.reset_index(drop=True),
            X_ai_test.reset_index(drop=True),
            y_ai_test.reset_index(drop=True),
            categorical=[AI4I_TYPE],
            numeric=AI4I_NUMERIC,
        ),
        "Scania": DatasetBundle(
            "Scania",
            X_sc_train.reset_index(drop=True),
            y_sc_train.reset_index(drop=True),
            X_sc_test.reset_index(drop=True),
            y_sc_test.reset_index(drop=True),
            categorical=[],
            numeric=list(X_sc_train.columns),
        ),
    }


def class_summary(y: pd.Series) -> Dict[str, float]:
    n = len(y)
    pos = int(y.sum())
    neg = int(n - pos)
    return {
        "n": n,
        "negative": neg,
        "positive": pos,
        "positive_rate": pos / n if n else np.nan,
        "neg_to_pos": neg / pos if pos else np.inf,
    }


def run_eda(bundles: Dict[str, DatasetBundle], out_dir: Path) -> None:
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in bundles.values():
        for split_name, X, y in [
            ("train", ds.X_train, ds.y_train),
            ("test", ds.X_test, ds.y_test),
        ]:
            cs = class_summary(y)
            rows.append({
                "dataset": ds.name,
                "split": split_name,
                "rows": len(X),
                "features": X.shape[1],
                "missing_cells": int(X.isna().sum().sum()),
                "missing_share": float(X.isna().mean().mean()),
                **cs,
            })
    overview = pd.DataFrame(rows)
    overview.to_csv(eda_dir / "dataset_overview.csv", index=False, encoding="utf-8-sig")

    # AI4I descriptive statistics and class-conditional summaries.
    ai = bundles["AI4I"]
    ai_full_X = pd.concat([ai.X_train, ai.X_test], ignore_index=True)
    ai_full_y = pd.concat([ai.y_train, ai.y_test], ignore_index=True)
    ai_full = ai_full_X.copy()
    ai_full[AI4I_TARGET] = ai_full_y.values
    ai_full.groupby(AI4I_TYPE)[AI4I_TARGET].agg(["count", "sum", "mean"]).rename(
        columns={"sum": "failures", "mean": "failure_rate"}
    ).to_csv(eda_dir / "ai4i_failure_rate_by_type.csv", encoding="utf-8-sig")
    ai_full.groupby(AI4I_TARGET)[AI4I_NUMERIC].agg(["mean", "std", "median"]).to_csv(
        eda_dir / "ai4i_numeric_by_class.csv", encoding="utf-8-sig"
    )

    # Scania missingness on TRAIN only; this is descriptive and cannot leak test labels.
    sc = bundles["Scania"]
    miss = pd.DataFrame({
        "feature": sc.X_train.columns,
        "missing_rate": sc.X_train.isna().mean().values,
        "missing_rate_pos": sc.X_train[sc.y_train.values == 1].isna().mean().values,
        "missing_rate_neg": sc.X_train[sc.y_train.values == 0].isna().mean().values,
    })
    miss["class_gap_abs"] = (miss["missing_rate_pos"] - miss["missing_rate_neg"]).abs()
    miss.sort_values(["class_gap_abs", "missing_rate"], ascending=False).to_csv(
        eda_dir / "scania_missingness_by_class.csv", index=False, encoding="utf-8-sig"
    )

    # Figures: one plot per figure, intentionally simple and reproducible.
    plot_df = overview.copy()
    plot_df["label"] = plot_df["dataset"] + " " + plot_df["split"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(plot_df["label"], plot_df["positive_rate"] * 100)
    ax.set_ylabel("Positive class, %")
    ax.set_title("Class imbalance by dataset split")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(eda_dir / "class_balance.png", dpi=180)
    plt.close(fig)

    top_miss = miss.sort_values("missing_rate", ascending=False).head(20).sort_values("missing_rate")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top_miss["feature"], top_miss["missing_rate"] * 100)
    ax.set_xlabel("Missing values, %")
    ax.set_title("Scania train: top-20 features by missingness")
    fig.tight_layout()
    fig.savefig(eda_dir / "scania_missingness_top20.png", dpi=180)
    plt.close(fig)


def make_ai4i_preprocessor() -> Tuple[ColumnTransformer, ColumnTransformer]:
    # Stage 1: categorical -> ordinal code; numeric -> standardized.
    # This representation is used by SMOTENC so Type stays categorical.
    pre = ColumnTransformer(
        transformers=[
            ("type", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), [AI4I_TYPE]),
            ("num", StandardScaler(), AI4I_NUMERIC),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    # Stage 2: one-hot Type after resampling; continuous features pass through.
    post = ColumnTransformer(
        transformers=[
            ("type_onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0]),
            ("num", "passthrough", list(range(1, 1 + len(AI4I_NUMERIC)))),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    return pre, post


def make_model(model_name: str, treatment: str, y_train: pd.Series):
    weighted = treatment == "weighting"
    if model_name == "logreg":
        return LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=3000,
            class_weight="balanced" if weighted else None,
            random_state=RANDOM_STATE,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced" if weighted else None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "xgb":
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        ratio = neg / pos
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=ratio if weighted else 1.0,
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist",
        )
    raise ValueError(model_name)


def make_pipeline(ds: DatasetBundle, model_name: str, treatment: str):
    model = make_model(model_name, treatment, ds.y_train)
    steps = []
    if ds.name == "AI4I":
        pre, post = make_ai4i_preprocessor()
        steps.append(("pre", pre))
        if treatment == "ros":
            steps.append(("sampler", RandomOverSampler(random_state=RANDOM_STATE)))
        elif treatment == "smote":
            steps.append(("sampler", SMOTENC(categorical_features=[0], random_state=RANDOM_STATE, k_neighbors=5)))
        steps.append(("post", post))
    else:
        # Use named columns rather than a hard-coded number if a mirror has the same 171 attributes in another order.
        pre = ColumnTransformer(
            transformers=[
                ("num", ImbPipeline([
                    ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
                    ("scaler", StandardScaler()),
                ]), ds.numeric),
            ],
            remainder="drop",
            sparse_threshold=0,
            verbose_feature_names_out=False,
        )
        steps.append(("pre", pre))
        if treatment == "ros":
            steps.append(("sampler", RandomOverSampler(random_state=RANDOM_STATE)))
        elif treatment == "smote":
            steps.append(("sampler", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def probabilities(estimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    score = estimator.decision_function(X)
    return 1 / (1 + np.exp(-score))


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "AP": float(average_precision_score(y_true, prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost_10fp_500fn": float(10 * fp + 500 * fn),
    }


def select_cost_threshold(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    """Choose threshold on training OOF predictions, never on test."""
    # Quantile grid is stable and avoids evaluating tens of thousands of duplicate score values.
    grid = np.unique(np.quantile(prob, np.linspace(0.001, 0.999, 1000)))
    best_t, best_cost = 0.5, math.inf
    for t in grid:
        pred = prob >= t
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = 10 * fp + 500 * fn
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost


def run_config(ds: DatasetBundle, model_name: str, treatment: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    oof = np.full(len(ds.y_train), np.nan, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(ds.X_train, ds.y_train), start=1):
        pipe = make_pipeline(ds, model_name, treatment)
        X_tr, y_tr = ds.X_train.iloc[tr_idx], ds.y_train.iloc[tr_idx]
        X_va, y_va = ds.X_train.iloc[va_idx], ds.y_train.iloc[va_idx]
        pipe.fit(X_tr, y_tr)
        p = probabilities(pipe, X_va)
        oof[va_idx] = p
        m = binary_metrics(y_va.to_numpy(), p, threshold=0.5)
        fold_rows.append({
            "dataset": ds.name,
            "model": model_name,
            "treatment": treatment,
            "fold": fold,
            **m,
        })

    # Threshold selection for Scania operational cost is based on OOF train predictions only.
    cost_threshold, oof_min_cost = select_cost_threshold(ds.y_train.to_numpy(), oof)

    final_pipe = make_pipeline(ds, model_name, treatment)
    final_pipe.fit(ds.X_train, ds.y_train)
    test_prob = probabilities(final_pipe, ds.X_test)
    test_default = binary_metrics(ds.y_test.to_numpy(), test_prob, threshold=0.5)
    test_cost_opt = binary_metrics(ds.y_test.to_numpy(), test_prob, threshold=cost_threshold)

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "dataset": ds.name,
        "model": model_name,
        "treatment": treatment,
        "cv_AP_mean": float(fold_df["AP"].mean()),
        "cv_AP_sd": float(fold_df["AP"].std(ddof=1)),
        "cv_precision_mean": float(fold_df["precision"].mean()),
        "cv_recall_mean": float(fold_df["recall"].mean()),
        "cv_f1_mean": float(fold_df["f1"].mean()),
        "test_AP": test_default["AP"],
        "test_precision_05": test_default["precision"],
        "test_recall_05": test_default["recall"],
        "test_f1_05": test_default["f1"],
        "test_fp_05": test_default["fp"],
        "test_fn_05": test_default["fn"],
        "test_cost_05": test_default["cost_10fp_500fn"],
        "cost_threshold_oof": cost_threshold,
        "oof_cost_at_selected_threshold": oof_min_cost,
        "test_fp_costopt": test_cost_opt["fp"],
        "test_fn_costopt": test_cost_opt["fn"],
        "test_cost_costopt": test_cost_opt["cost_10fp_500fn"],
        "test_prob": test_prob,
    }
    return fold_df, summary


def normalized_gain(ap: float, baseline_ap: float) -> float:
    denom = 1.0 - baseline_ap
    if denom <= 0:
        return np.nan
    return (ap - baseline_ap) / denom


def stability_tables(summary_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset in ["AI4I", "Scania"]:
        ddf = summary_df[summary_df["dataset"] == dataset]
        for model in MODELS:
            mdf = ddf[ddf["model"] == model].set_index("treatment")
            base = float(mdf.loc["baseline", "test_AP"])
            for treatment in ["weighting", "ros", "smote"]:
                ap = float(mdf.loc[treatment, "test_AP"])
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "treatment": treatment,
                    "baseline_AP": base,
                    "AP": ap,
                    "delta_AP": ap - base,
                    "G": normalized_gain(ap, base),
                })
    gains = pd.DataFrame(rows)

    d_rows = []
    for model in MODELS:
        for treatment in ["weighting", "ros", "smote"]:
            a = gains[(gains.dataset == "AI4I") & (gains.model == model) & (gains.treatment == treatment)].iloc[0]
            s = gains[(gains.dataset == "Scania") & (gains.model == model) & (gains.treatment == treatment)].iloc[0]
            d_rows.append({
                "model": model,
                "treatment": treatment,
                "D": abs(float(a.G) - float(s.G)),
                "G_AI4I": float(a.G),
                "G_Scania": float(s.G),
            })
    ddf = pd.DataFrame(d_rows)
    s_df = ddf.groupby("treatment", as_index=False)["D"].mean().rename(columns={"D": "S_mean_D"})
    return gains, ddf.merge(s_df, on="treatment", how="left")


def bootstrap_stability(
    bundles: Dict[str, DatasetBundle],
    prob_store: Dict[Tuple[str, str, str], np.ndarray],
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Paired bootstrap within each dataset; same resampled rows are shared by all configs."""
    rng = np.random.default_rng(RANDOM_STATE)
    records = []
    for b in range(n_boot):
        sample_idx = {
            ds_name: rng.integers(0, len(ds.y_test), len(ds.y_test))
            for ds_name, ds in bundles.items()
        }
        g = {}
        valid = True
        for ds_name, ds in bundles.items():
            idx = sample_idx[ds_name]
            yb = ds.y_test.to_numpy()[idx]
            if np.unique(yb).size < 2:
                valid = False
                break
            for model in MODELS:
                base_prob = prob_store[(ds_name, model, "baseline")][idx]
                base_ap = average_precision_score(yb, base_prob)
                for treatment in ["weighting", "ros", "smote"]:
                    p = prob_store[(ds_name, model, treatment)][idx]
                    ap = average_precision_score(yb, p)
                    g[(ds_name, model, treatment)] = normalized_gain(ap, base_ap)
        if not valid:
            continue
        S = {}
        for treatment in ["weighting", "ros", "smote"]:
            Ds = [abs(g[("AI4I", m, treatment)] - g[("Scania", m, treatment)]) for m in MODELS]
            S[treatment] = float(np.mean(Ds))
        records.append({
            "boot": b,
            "S_weighting": S["weighting"],
            "S_ros": S["ros"],
            "S_smote": S["smote"],
            # Confirmatory contrasts are defined exactly as in H1:
            # weighting must be LOWER than the competing method, so the contrast must be < 0.
            "contrast_weighting_minus_ros": S["weighting"] - S["ros"],
            "contrast_weighting_minus_smote": S["weighting"] - S["smote"],
        })
    return pd.DataFrame(records)


def bootstrap_summary(boot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (descriptive intervals, confirmatory H1 table).

    Confirmatory rule is pre-specified in the Research Proposal:
    for both weighting-vs-ROS and weighting-vs-SMOTE, the contrast
    S_weighting - S_competitor must be negative and the upper bound of the
    one-sided 97.5% bootstrap interval must be below zero. The 97.5% level
    implements Bonferroni correction for two confirmatory comparisons at
    family-wise alpha=0.05.
    """
    desc_rows = []
    for col in [
        "S_weighting", "S_ros", "S_smote",
        "contrast_weighting_minus_ros", "contrast_weighting_minus_smote",
    ]:
        s = boot[col].dropna()
        desc_rows.append({
            "statistic": col,
            "bootstrap_mean": float(s.mean()),
            "two_sided_95_low": float(s.quantile(0.025)),
            "two_sided_95_high": float(s.quantile(0.975)),
        })
    descriptive = pd.DataFrame(desc_rows)

    confirmatory_rows = []
    for competitor, col in [
        ("ros", "contrast_weighting_minus_ros"),
        ("smote", "contrast_weighting_minus_smote"),
    ]:
        s = boot[col].dropna()
        upper = float(s.quantile(0.975))
        estimate = float(s.mean())
        confirmatory_rows.append({
            "comparison": f"weighting_vs_{competitor}",
            "contrast": "S_weighting - S_competitor",
            "bootstrap_mean_contrast": estimate,
            "one_sided_upper_97_5": upper,
            "criterion": "contrast < 0 and upper_97.5% < 0",
            "supported": bool((estimate < 0.0) and (upper < 0.0)),
        })
    confirmatory = pd.DataFrame(confirmatory_rows)
    return descriptive, confirmatory


def save_environment(out_dir: Path) -> None:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "xgboost": xgboost.__version__,
        "random_state": RANDOM_STATE,
        "cv_splits": N_SPLITS,
    }
    (out_dir / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 2 experiment: class-imbalance robustness on AI4I and Scania APS")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_chapter2"))
    parser.add_argument("--download", action="store_true", help="Download the two official UCI archives if local files are missing")
    parser.add_argument("--eda-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    set_seed()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_data(args.data_dir, args.download)
    bundles = build_bundles(args.data_dir)
    run_eda(bundles, args.out_dir)
    save_environment(args.out_dir)

    if args.eda_only:
        print(f"EDA completed: {args.out_dir / 'eda'}")
        return

    all_folds = []
    summaries = []
    prob_store: Dict[Tuple[str, str, str], np.ndarray] = {}

    total = len(bundles) * len(MODELS) * len(TREATMENTS)
    done = 0
    for ds_name, ds in bundles.items():
        for model in MODELS:
            for treatment in TREATMENTS:
                done += 1
                print(f"[{done}/{total}] {ds_name} | {model} | {treatment}", flush=True)
                fold_df, summary = run_config(ds, model, treatment)
                all_folds.append(fold_df)
                prob_store[(ds_name, model, treatment)] = summary.pop("test_prob")
                summaries.append(summary)

    fold_df = pd.concat(all_folds, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    fold_df.to_csv(args.out_dir / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.out_dir / "test_summary.csv", index=False, encoding="utf-8-sig")

    gains, stability = stability_tables(summary_df)
    gains.to_csv(args.out_dir / "normalized_gains.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(args.out_dir / "stability_D_S.csv", index=False, encoding="utf-8-sig")

    if args.bootstrap > 0:
        boot = bootstrap_stability(bundles, prob_store, n_boot=args.bootstrap)
        boot.to_csv(args.out_dir / "bootstrap_stability.csv", index=False, encoding="utf-8-sig")
        descriptive_ci, confirmatory = bootstrap_summary(boot)
        descriptive_ci.to_csv(args.out_dir / "bootstrap_descriptive_ci.csv", index=False, encoding="utf-8-sig")
        confirmatory.to_csv(args.out_dir / "bootstrap_confirmatory_H1.csv", index=False, encoding="utf-8-sig")

    # Compact manifest for reproducibility.
    manifest = {
        "configurations": total,
        "models": MODELS,
        "treatments": TREATMENTS,
        "datasets": list(bundles),
        "primary_metric": "Average Precision",
        "stability": "G=(AP_treatment-AP_baseline)/(1-AP_baseline); D=|G_AI4I-G_Scania|; S=mean_model(D)",
        "scania_cost": "10*FP + 500*FN; threshold selected on training OOF probabilities only",
        "test_policy": "AI4I 20% stratified holdout; Scania original UCI test set",
        "confirmatory_H1": "Both S_weighting-S_ROS and S_weighting-S_SMOTE must be <0 with one-sided 97.5% bootstrap upper bound <0 (Bonferroni, family alpha=0.05)",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
