"""Train-only missing values, categorical encoding, and a logistic baseline."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from data_preparation import load_loans, prepare_loans, temporal_split
from features import MTHS_SINCE_FILL, prepare_model_frames


def main():
    print("Loading and preparing loans...")
    df = prepare_loans(load_loans())
    train_df, test_df = temporal_split(df)

    (
        X_train,
        y_train,
        (X_test,),
        (y_test,),
        groups,
        preprocessor,
        flag_columns,
    ) = prepare_model_frames(train_df, test_df)

    numeric, mths_since, flags, categorical = groups

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Numeric features: {len(numeric)}")
    print(f"mths_since features (fill={MTHS_SINCE_FILL}): {len(mths_since)}")
    print(f"Missing flags: {len(flags)}")
    print(f"Categorical features: {categorical}")
    if flags:
        print("Flags created for:")
        print(", ".join(flag_columns))

    pipeline = Pipeline(
        [
            ("features", preprocessor),
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

    print("\nFitting logistic regression on pre-2018 loans...")
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\nBaseline results on 2018+ test loans")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, proba):.4f}")
    print(f"Positive rate in test: {y_test.mean():.4f}")
    print("\nClassification report (threshold 0.5):")
    print(classification_report(y_test, pred, digits=3))


if __name__ == "__main__":
    main()
