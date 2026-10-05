"""Compare logistic vs XGBoost, with pricing ablation and probability calibration."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

from data_preparation import load_loans, prepare_loans
from features import LC_PRICING_COLUMNS, prepare_model_frames

RANDOM_STATE = 42
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def three_way_split(df):
    train_df = df[df["issue_d"] < "2017-01-01"].copy()
    valid_df = df[
        (df["issue_d"] >= "2017-01-01") & (df["issue_d"] < "2018-01-01")
    ].copy()
    test_df = df[df["issue_d"] >= "2018-01-01"].copy()
    return train_df, valid_df, test_df


def metrics(y_true, proba, label):
    return {
        "model": label,
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "mean_pd": float(np.mean(proba)),
        "actual_default_rate": float(np.mean(y_true)),
    }


def print_metrics(row, split_name):
    print(
        f"  {split_name:<5} {row['model']:<28} "
        f"ROC-AUC {row['roc_auc']:.4f}  "
        f"PR-AUC {row['pr_auc']:.4f}  "
        f"Brier {row['brier']:.4f}  "
        f"mean PD {row['mean_pd']:.4f} vs actual {row['actual_default_rate']:.4f}"
    )


def calibrate(y_valid, valid_proba, test_proba):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(valid_proba, y_valid)
    return iso.transform(test_proba), iso


def top_xgb_importances(model, preprocessor, n=15):
    names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:n]
    return pd.DataFrame(
        {"feature": names[order], "importance": importances[order]}
    )


def train_feature_set(name, train_df, valid_df, test_df, extra_drop=None):
    print(f"\n=== {name} ===")
    (
        X_train,
        y_train,
        (X_valid, X_test),
        (y_valid, y_test),
        groups,
        preprocessor,
        _,
    ) = prepare_model_frames(train_df, valid_df, test_df, extra_drop=extra_drop)

    numeric, mths_since, flags, categorical = groups
    print(
        f"Rows train/valid/test: {len(X_train):,} / {len(X_valid):,} / {len(X_test):,}"
    )
    print(
        f"Features numeric={len(numeric)} mths_since={len(mths_since)} "
        f"flags={len(flags)} categorical={categorical}"
    )

    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_valid_t = preprocessor.transform(X_valid)
    X_test_t = preprocessor.transform(X_test)

    results = []

    logistic = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    print("Fitting logistic regression...")
    logistic.fit(X_train_t, y_train)
    logistic_test = logistic.predict_proba(X_test_t)[:, 1]
    logistic_valid = logistic.predict_proba(X_valid_t)[:, 1]
    row = metrics(y_test, logistic_test, f"{name} | logistic")
    results.append(row)
    print_metrics(metrics(y_valid, logistic_valid, f"{name} | logistic"), "valid")
    print_metrics(row, "test")

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        early_stopping_rounds=40,
        scale_pos_weight=neg / pos,
    )
    print("Fitting XGBoost with early stopping on 2017...")
    xgb.fit(
        X_train_t,
        y_train,
        eval_set=[(X_valid_t, y_valid)],
        verbose=False,
    )
    best_iter = int(getattr(xgb, "best_iteration", xgb.n_estimators - 1))
    print(f"XGBoost best iteration: {best_iter}")

    xgb_valid = xgb.predict_proba(X_valid_t)[:, 1]
    xgb_test = xgb.predict_proba(X_test_t)[:, 1]
    row = metrics(y_test, xgb_test, f"{name} | xgboost")
    results.append(row)
    print_metrics(metrics(y_valid, xgb_valid, f"{name} | xgboost"), "valid")
    print_metrics(row, "test")

    xgb_test_cal, calibrator = calibrate(y_valid, xgb_valid, xgb_test)
    row = metrics(y_test, xgb_test_cal, f"{name} | xgboost+isotonic")
    results.append(row)
    print_metrics(row, "test")

    print("\nTop XGBoost importances:")
    print(top_xgb_importances(xgb, preprocessor).to_string(index=False))

    return {
        "results": results,
        "preprocessor": preprocessor,
        "xgb": xgb,
        "calibrator": calibrator,
        "y_test": y_test,
        "xgb_test": xgb_test,
        "xgb_test_cal": xgb_test_cal,
    }


def main():
    print("Loading and preparing loans...")
    df = prepare_loans(load_loans())
    train_df, valid_df, test_df = three_way_split(df)

    print("\nSplit sizes")
    print(f"  train <2017: {len(train_df):,}  default rate {train_df['default_flag'].mean():.4f}")
    print(f"  valid 2017:  {len(valid_df):,}  default rate {valid_df['default_flag'].mean():.4f}")
    print(f"  test  2018+: {len(test_df):,}  default rate {test_df['default_flag'].mean():.4f}")

    full = train_feature_set("full features", train_df, valid_df, test_df)
    ablation = train_feature_set(
        "no LC pricing",
        train_df,
        valid_df,
        test_df,
        extra_drop=LC_PRICING_COLUMNS,
    )

    print("\n=== Test-set comparison (2018+) ===")
    all_rows = full["results"] + ablation["results"]
    summary = pd.DataFrame(all_rows)
    print(
        summary.to_string(
            index=False,
            formatters={
                "roc_auc": "{:.4f}".format,
                "pr_auc": "{:.4f}".format,
                "brier": "{:.4f}".format,
                "mean_pd": "{:.4f}".format,
                "actual_default_rate": "{:.4f}".format,
            },
        )
    )

    MODELS_DIR.mkdir(exist_ok=True)
    np.savez(
        MODELS_DIR / "xgb_full_test_scores.npz",
        y_test=full["y_test"].to_numpy(),
        pd_raw=full["xgb_test"],
        pd_calibrated=full["xgb_test_cal"],
    )
    print(f"\nSaved test scores to {MODELS_DIR / 'xgb_full_test_scores.npz'}")


if __name__ == "__main__":
    main()
