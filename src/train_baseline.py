"""Train-only missing values, categorical encoding, and a logistic baseline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preparation import load_loans, prepare_loans, temporal_split

TARGET = "default_flag"

# Not available at a useful rate, collinear, or not a credit feature.
DROP_COLUMNS = [
    "issue_d",
    "zip_code",
    "funded_amnt",
    "funded_amnt_inv",
    "sub_grade",
    "initial_list_status",
    "fico_range_low",
    "fico_range_high",
    "annual_inc_joint",
    "dti_joint",
    "verification_status_joint",
    "revol_bal_joint",
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
    "sec_credit_history_months",
]

CATEGORICAL_COLUMNS = [
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
]

MISSING_FLAG_THRESHOLD = 0.05
# "Never happened" for delinquency recency fields.
MTHS_SINCE_FILL = 999


def add_missing_flags(train_df, test_df, columns, threshold=MISSING_FLAG_THRESHOLD):
    """Create missing flags from train missing rates, then apply to both splits."""
    missing_rate = train_df[columns].isna().mean()
    flag_columns = missing_rate[missing_rate >= threshold].index.tolist()

    train_out = train_df.copy()
    test_out = test_df.copy()
    for column in flag_columns:
        flag_name = f"{column}_missing"
        train_out[flag_name] = train_out[column].isna().astype(int)
        test_out[flag_name] = test_out[column].isna().astype(int)

    return train_out, test_out, flag_columns


def split_feature_groups(df):
    categorical = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    mths_since = [
        c
        for c in df.columns
        if c.startswith("mths_since_") and not c.endswith("_missing")
    ]
    flag_columns = [c for c in df.columns if c.endswith("_missing")]
    numeric = [
        c
        for c in df.columns
        if c not in categorical
        and c not in mths_since
        and c not in flag_columns
        and c != TARGET
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric, mths_since, flag_columns, categorical


def build_pipeline(numeric, mths_since, flag_columns, categorical):
    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if mths_since:
        transformers.append(
            (
                "mths_since",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value=MTHS_SINCE_FILL
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                mths_since,
            )
        )

    if flag_columns:
        transformers.append(("missing_flags", "passthrough", flag_columns))

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="first",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    return Pipeline(
        [
            ("features", ColumnTransformer(transformers, remainder="drop")),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def main():
    print("Loading and preparing loans...")
    df = prepare_loans(load_loans())
    train_df, test_df = temporal_split(df)

    train_df = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
    test_df = test_df.drop(columns=DROP_COLUMNS, errors="ignore")

    candidate_numeric = [
        c
        for c in train_df.columns
        if c != TARGET and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    train_df, test_df, flag_columns = add_missing_flags(
        train_df, test_df, candidate_numeric
    )

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)
    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET])

    numeric, mths_since, flags, categorical = split_feature_groups(X_train)

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Numeric features: {len(numeric)}")
    print(f"mths_since features (fill={MTHS_SINCE_FILL}): {len(mths_since)}")
    print(f"Missing flags: {len(flags)}")
    print(f"Categorical features: {categorical}")
    if flags:
        print("Flags created for:")
        print(", ".join(flag_columns))

    pipeline = build_pipeline(numeric, mths_since, flags, categorical)

    print("\nFitting logistic regression on pre-2018 loans...")
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    print("\nBaseline results on 2018+ test loans")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Positive rate in test: {y_test.mean():.4f}")
    print("\nClassification report (threshold 0.5):")
    print(classification_report(y_test, pred, digits=3))


if __name__ == "__main__":
    main()
