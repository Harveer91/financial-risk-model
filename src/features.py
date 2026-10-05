import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default_flag"

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

# LC pricing fields: keep in the main model, drop for the ablation.
LC_PRICING_COLUMNS = ["grade", "int_rate", "installment"]

CATEGORICAL_COLUMNS = [
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
]

MISSING_FLAG_THRESHOLD = 0.05
MTHS_SINCE_FILL = 999


def add_missing_flags(train_df, other_frames, columns, threshold=MISSING_FLAG_THRESHOLD):
    """Create missing flags from train missing rates, then apply to every split."""
    missing_rate = train_df[columns].isna().mean()
    flag_columns = missing_rate[missing_rate >= threshold].index.tolist()

    def apply_flags(df):
        out = df.copy()
        for column in flag_columns:
            out[f"{column}_missing"] = out[column].isna().astype(int)
        return out

    train_out = apply_flags(train_df)
    others_out = [apply_flags(df) for df in other_frames]
    return train_out, others_out, flag_columns


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


def build_preprocessor(numeric, mths_since, flag_columns, categorical):
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

    return ColumnTransformer(transformers, remainder="drop")


def prepare_model_frames(train_df, *other_frames, extra_drop=None):
    drop_cols = list(DROP_COLUMNS)
    if extra_drop:
        drop_cols.extend(extra_drop)

    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    others = [df.drop(columns=drop_cols, errors="ignore") for df in other_frames]

    candidate_numeric = [
        c
        for c in train_df.columns
        if c != TARGET and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    train_df, others, flag_columns = add_missing_flags(
        train_df, others, candidate_numeric
    )

    y_train = train_df[TARGET].astype(int)
    X_train = train_df.drop(columns=[TARGET])
    y_others = [df[TARGET].astype(int) for df in others]
    X_others = [df.drop(columns=[TARGET]) for df in others]

    groups = split_feature_groups(X_train)
    preprocessor = build_preprocessor(*groups)
    return X_train, y_train, X_others, y_others, groups, preprocessor, flag_columns
