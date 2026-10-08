"""
train_and_save.py
Fit a churn-prediction Logistic Regression pipeline on the Main Sample dataset
(target: `Exited`), excluding the leaky `Complain` column. Dump artefacts for
the Gradio what-if calculator.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SOURCE_CSV = Path(
    r"C:\Users\khali\iCloudDrive\Workspace\UNIVERSITIES\NOROFF\YEAR 2\Statistical Analysis Tools and Techniques\Assignment 3\Main Sample.csv"
)
ART = Path(__file__).parent / "artifacts"
ART.mkdir(exist_ok=True)
SEED = 42
N_SAMPLE_CUSTOMERS = 50

# ───────────────────────── load + clean ─────────────────────────
print(f"Loading {SOURCE_CSV.name}...")
df = pd.read_csv(SOURCE_CSV)
# Drop trailing unnamed empty columns (CSV export artefact)
df = df.loc[:, ~df.columns.str.match(r"Unnamed:")].copy()
df.columns = [c.strip() for c in df.columns]
df = df.dropna(how="all").reset_index(drop=True)
print(f"  raw shape: {df.shape}  cols: {list(df.columns)}")

TARGET = "Exited"
DROP_COLS = ["CustomerId", "Complain", TARGET]
NUM_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "EstimatedSalary", "Satisfaction Score", "Point Earned"]
CAT_COLS = ["Location", "Gender", "HasCreditCard", "IsActiveMember", "Card Type"]

# Sanity: ensure all expected columns exist.
for c in NUM_COLS + CAT_COLS + [TARGET]:
    if c not in df.columns:
        raise RuntimeError(f"Missing expected column: {c}")

X = df[NUM_COLS + CAT_COLS].copy()
y = df[TARGET].astype(int)
print(f"  class balance: {dict(y.value_counts())}")

# ───────────────────────── pipeline ─────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUM_COLS),
    ("cat", categorical_transformer, CAT_COLS),
])
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
])

# ───────────────────────── 5-fold stratified CV ─────────────────────────
print("\n5-fold stratified CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv = cross_validate(pipe, X, y, cv=skf,
                    scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1)
metrics = {
    "n_samples": int(len(X)),
    "n_folds": 5,
    "accuracy": {"mean": float(cv["test_accuracy"].mean()),
                 "std": float(cv["test_accuracy"].std())},
    "f1": {"mean": float(cv["test_f1"].mean()),
           "std": float(cv["test_f1"].std())},
    "roc_auc": {"mean": float(cv["test_roc_auc"].mean()),
                "std": float(cv["test_roc_auc"].std())},
}
for k in ["accuracy", "f1", "roc_auc"]:
    print(f"  {k:<8s}: {metrics[k]['mean']:.4f} ± {metrics[k]['std']:.4f}")

# ───────────────────────── final fit on full data ─────────────────────────
print("\nFitting on full data...")
pipe.fit(X, y)
joblib.dump(pipe, ART / "pipeline.joblib")

# ───────────────────────── interpretability data ─────────────────────────
ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_expanded = list(ohe.get_feature_names_out(CAT_COLS))
all_features = list(NUM_COLS) + cat_expanded
coefs = pipe.named_steps["classifier"].coef_[0]
intercept = float(pipe.named_steps["classifier"].intercept_[0])
interp = {
    "feature_names": all_features,
    "coefficients": [float(c) for c in coefs],
    "intercept": intercept,
}
(ART / "interpretability.json").write_text(json.dumps(interp, indent=2))

# ───────────────────────── sample customers ─────────────────────────
# Stratified sample of N customers (half exited, half stayed) so the
# dropdown has interesting examples on both sides.
print(f"\nSampling {N_SAMPLE_CUSTOMERS} customers (stratified by Exited)...")
rng = np.random.RandomState(SEED)
n_each = N_SAMPLE_CUSTOMERS // 2
exited = df[df[TARGET] == 1].sample(n=n_each, random_state=SEED).reset_index(drop=True)
stayed = df[df[TARGET] == 0].sample(n=n_each, random_state=SEED).reset_index(drop=True)
sample = pd.concat([exited, stayed], ignore_index=True)
# Add the model's predicted churn probability so the dropdown can rank by interest.
sample_probs = pipe.predict_proba(sample[NUM_COLS + CAT_COLS])[:, 1]
sample = sample.assign(_predicted_churn_proba=sample_probs)
sample.to_csv(ART / "sample_customers.csv", index=False)

# ───────────────────────── metadata ─────────────────────────
num_stats = {col: {"min": float(X[col].min()),
                   "max": float(X[col].max()),
                   "median": float(X[col].median()),
                   "step": 1.0 if X[col].dtype.kind in "iu" else 0.01}
             for col in NUM_COLS}
cat_choices = {col: sorted(map(str, df[col].dropna().unique().tolist())) for col in CAT_COLS}

meta = {
    "num_cols": NUM_COLS,
    "cat_cols": CAT_COLS,
    "num_stats": num_stats,
    "cat_choices": cat_choices,
    "target": TARGET,
    "n_train_rows": int(len(X)),
    "feature_order": NUM_COLS + CAT_COLS,
}
(ART / "meta.json").write_text(json.dumps(meta, indent=2))
(ART / "metrics.json").write_text(json.dumps(metrics, indent=2))

print("\nArtefact sizes:")
for p in sorted(ART.iterdir()):
    print(f"  {p.name}: {p.stat().st_size / 1024:.1f} KB")
print("\nDone.")
