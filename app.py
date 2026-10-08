"""
app.py: Bank Customer Churn what-if calculator.

Pick a real customer → their attributes populate the inputs → adjust any input
and the predicted churn probability + driver breakdown update live (no Predict
button). The reactivity is the point of the UX.
"""

import json
import tempfile
from pathlib import Path

import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ART = Path(__file__).parent / "artifacts"

# ───────────────────────── load artefacts ─────────────────────────
print("Loading artefacts...")
pipe = joblib.load(ART / "pipeline.joblib")
meta = json.loads((ART / "meta.json").read_text())
metrics = json.loads((ART / "metrics.json").read_text())
interp = json.loads((ART / "interpretability.json").read_text())
sample = pd.read_csv(ART / "sample_customers.csv")

NUM_COLS = meta["num_cols"]
CAT_COLS = meta["cat_cols"]
NUM_STATS = meta["num_stats"]
CAT_CHOICES = meta["cat_choices"]
FEATURE_ORDER = meta["feature_order"]
TARGET = meta["target"]

FEATURE_NAMES = interp["feature_names"]
COEFS = np.array(interp["coefficients"])

print(f"Loaded: pipeline + {len(sample)} sample customers.")


# ───────────────────────── helpers ─────────────────────────
def _build_row(*values):
    """Map flat list of input values to a 1-row DataFrame in FEATURE_ORDER."""
    row = dict(zip(FEATURE_ORDER, values))
    df = pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    # Numeric casts (sliders return floats; OHE doesn't care but scaler does).
    for c in NUM_COLS:
        df[c] = df[c].astype(float)
    return df


def _grouped_contributions(row_df):
    """Compute per-original-feature contributions (coef × scaled value),
    aggregating OHE'd columns back to their parent feature name.
    Returns a list of (feature, contribution) sorted by |contribution|.
    """
    # Run preprocessor only.
    X_processed = pipe.named_steps["preprocessor"].transform(row_df)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = X_processed[0]
    contribs = COEFS * X_processed  # element-wise

    # Group by original feature: numerics map 1:1, categoricals fold all OHE cols.
    parents = []
    for f in FEATURE_NAMES:
        if "_" in f and f.split("_", 1)[0] in CAT_COLS:
            parents.append(f.split("_", 1)[0])
        elif f in CAT_COLS:
            parents.append(f)
        else:
            parents.append(f)
    grouped = {}
    for p, c in zip(parents, contribs):
        grouped[p] = grouped.get(p, 0.0) + c

    items = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return items


def _risk_band(p):
    if p >= 0.5:
        return ("Likely to churn", "#ef4444")
    if p >= 0.25:
        return ("Elevated risk", "#f59e0b")
    return ("Likely to stay", "#10b981")


def _contrib_plot(contribs):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    feats = [f for f, _ in contribs][::-1]
    vals = [v for _, v in contribs][::-1]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in vals]
    ax.barh(feats, vals, color=colors)
    ax.axvline(0, color="#94a3b8", linewidth=0.7)
    ax.set_xlabel("Contribution to churn logit (red = pushes toward churn)")
    ax.set_title("Why this prediction: feature drivers")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


# ───────────────────────── core scoring ─────────────────────────
def score(*values):
    row_df = _build_row(*values)
    p = float(pipe.predict_proba(row_df)[0, 1])
    band, color = _risk_band(p)

    summary = (
        f"<div style='text-align:center;padding:1.2rem;border-radius:8px;"
        f"background:rgba(0,0,0,0.18);border-left:4px solid {color}'>"
        f"<div style='font-size:0.8rem;color:#94a3b8;font-family:monospace;"
        f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem'>"
        f"Predicted churn probability</div>"
        f"<div style='font-size:2.4rem;font-weight:700;color:{color};line-height:1'>"
        f"{p * 100:.1f}%</div>"
        f"<div style='margin-top:0.4rem;color:{color};font-weight:600'>{band}</div>"
        f"</div>"
    )

    contribs = _grouped_contributions(row_df)
    plot_path = _contrib_plot(contribs)
    return summary, plot_path


# ───────────────────────── load preset customer ─────────────────────────
def _label_for_customer(row):
    actual = "Exited" if int(row[TARGET]) == 1 else "Stayed"
    return f"[{actual}] Customer {int(row['CustomerId'])}, age {int(row['Age'])}, " \
           f"{row['Location']}, model says {row['_predicted_churn_proba'] * 100:.0f}%"


sample["_label"] = sample.apply(_label_for_customer, axis=1)
# Order: most interesting first, biggest disagreement between actual and prediction.
sample["_interest"] = (sample[TARGET] - sample["_predicted_churn_proba"]).abs()
sample = sample.sort_values("_interest", ascending=False).reset_index(drop=True)
CUSTOMER_LABELS = sample["_label"].tolist()


def load_customer(label):
    row = sample[sample["_label"] == label].iloc[0]
    return [row[c] for c in NUM_COLS] + [str(row[c]) for c in CAT_COLS]


# ───────────────────────── UI ─────────────────────────
HEAD = (
    "# Bank Customer Churn: What-If Calculator\n"
    "Pick a real customer from the dataset (or just tweak the inputs from scratch). "
    "The model re-scores live on every change, no Predict button. The driver panel "
    "below shows which inputs are currently pushing the prediction up or down for *this* "
    "customer."
)


def num_input(col):
    s = NUM_STATS[col]
    return gr.Slider(
        minimum=int(s["min"]),
        maximum=int(s["max"]),
        step=int(s["step"]),
        value=int(s["median"]),
        label=col,
    )


with gr.Blocks(theme=gr.themes.Soft(), title="Bank Customer Churn: What If") as demo:
    gr.Markdown(HEAD)

    customer_dd = gr.Dropdown(
        choices=CUSTOMER_LABELS,
        value=CUSTOMER_LABELS[0],
        label="Pick a real customer to start from",
        filterable=True,
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Customer attributes")
            with gr.Row():
                with gr.Column():
                    inp_credit = num_input("CreditScore")
                    inp_age = num_input("Age")
                    inp_tenure = num_input("Tenure")
                    inp_balance = num_input("Balance")
                with gr.Column():
                    inp_products = num_input("NumOfProducts")
                    inp_salary = num_input("EstimatedSalary")
                    inp_sat = num_input("Satisfaction Score")
                    inp_points = num_input("Point Earned")
            with gr.Row():
                inp_location = gr.Dropdown(CAT_CHOICES["Location"], label="Location",
                                           value=CAT_CHOICES["Location"][0])
                inp_gender = gr.Dropdown(CAT_CHOICES["Gender"], label="Gender",
                                         value=CAT_CHOICES["Gender"][0])
                inp_cc = gr.Dropdown(CAT_CHOICES["HasCreditCard"], label="Has credit card",
                                     value=CAT_CHOICES["HasCreditCard"][0])
                inp_active = gr.Dropdown(CAT_CHOICES["IsActiveMember"], label="Active member",
                                         value=CAT_CHOICES["IsActiveMember"][0])
                inp_card = gr.Dropdown(CAT_CHOICES["Card Type"], label="Card type",
                                       value=CAT_CHOICES["Card Type"][0])
        with gr.Column(scale=2):
            churn_card = gr.HTML()
            gr.Markdown(
                f"*Model: Logistic Regression, 5-fold stratified CV: "
                f"Accuracy {metrics['accuracy']['mean']:.3f} · "
                f"F1 {metrics['f1']['mean']:.3f} · "
                f"ROC-AUC {metrics['roc_auc']['mean']:.3f}*"
            )

    contrib_plot = gr.Image(label="Driver breakdown", type="filepath",
                            show_download_button=False, interactive=False)

    INPUTS = [
        inp_credit, inp_age, inp_tenure, inp_balance, inp_products,
        inp_salary, inp_sat, inp_points,
        inp_location, inp_gender, inp_cc, inp_active, inp_card,
    ]
    OUTPUTS = [churn_card, contrib_plot]

    # Wire customer picker → input values (which then triggers .change on each input below).
    customer_dd.change(load_customer, inputs=customer_dd, outputs=INPUTS)

    # Live re-score on any input change.
    for inp in INPUTS:
        inp.change(score, inputs=INPUTS, outputs=OUTPUTS)

    # Initial state: load the top customer on first render.
    demo.load(load_customer, inputs=customer_dd, outputs=INPUTS).then(
        score, inputs=INPUTS, outputs=OUTPUTS
    )

if __name__ == "__main__":
    demo.launch(show_api=False)
