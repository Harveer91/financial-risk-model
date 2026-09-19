import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

query = "SELECT * FROM clean_loans_model"

df = pd.read_sql(query, engine)

numeric_columns = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "mths_since_last_major_derog",
    "annual_inc_joint",
    "dti_joint",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",
    "total_rev_hi_lim",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
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
    "sec_app_mths_since_last_major_derog"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["issue_d"] = pd.to_datetime(
    df["issue_d"],
    format="%b-%y",
    errors="coerce"
)

# Convert loan term to number of months
df["term"] = df["term"].str.extract(r"(\d+)").astype(float)

# Convert employment length to years
df["emp_length"] = (
    df["emp_length"]
    .str.replace("+ years", "", regex=False)
    .str.replace("< 1 year", "0", regex=False)
    .str.replace(" years", "", regex=False)
    .str.replace(" year", "", regex=False)
)

df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")

# Convert credit history start date
df["earliest_cr_line"] = pd.to_datetime(
    df["earliest_cr_line"],
    format="%b-%y",
    errors="coerce"
)

# Calculate credit history length in months
df["credit_history_months"] = (
    (df["issue_d"].dt.year - df["earliest_cr_line"].dt.year) * 12
    + (df["issue_d"].dt.month - df["earliest_cr_line"].dt.month)
)

# Convert secondary applicant credit history date
df["sec_app_earliest_cr_line"] = pd.to_datetime(
    df["sec_app_earliest_cr_line"],
    format="%b-%y",
    errors="coerce"
)

df["sec_credit_history_months"] = (
    (df["issue_d"].dt.year - df["sec_app_earliest_cr_line"].dt.year) * 12
    + (df["issue_d"].dt.month - df["sec_app_earliest_cr_line"].dt.month)
)

df = df.drop(
    columns=["earliest_cr_line", "sec_app_earliest_cr_line"]
)

print("Data preparation successful.")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nData types:")
print(df.dtypes.value_counts())

print("\nTarget distribution:")
print(df["default_flag"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nCategorical columns:")
for column in df.select_dtypes(include=["str"]).columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).head(10))

print("\nIssue date range:")
print("Earliest:", df["issue_d"].min())
print("Latest:", df["issue_d"].max())

train_df = df[df["issue_d"] < "2018-01-01"].copy()
test_df = df[df["issue_d"] >= "2018-01-01"].copy()

print("\nTrain/test split:")
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")

print("\nTraining date range:")
print(train_df["issue_d"].min(), "to", train_df["issue_d"].max())

print("\nTesting date range:")
print(test_df["issue_d"].min(), "to", test_df["issue_d"].max())

print("\nTraining target distribution:")
print(train_df["default_flag"].value_counts())

print("\nTesting target distribution:")
print(test_df["default_flag"].value_counts())

print("\nConverted features:")
print(df[["term", "emp_length"]].head(10))

print("\nCredit history features:")
print("\nCredit history features:")
print(
    df[
        [
            "issue_d",
            "credit_history_months",
            "sec_credit_history_months"
        ]
    ].head(10)
)