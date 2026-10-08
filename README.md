---
title: "Bank Customer Churn: What-If Calculator"
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Bank Customer Churn: What-If Calculator

Logistic-regression churn model on a ~10 000-row European retail-banking dataset, wrapped in a live what-if calculator. Pick a real customer from the dataset (their features populate the inputs), then nudge any of those inputs and watch the churn probability update in real time. The right-hand panel shows which features are pushing the prediction toward or away from churn.

## What this demo does

- **Pick a real customer:** a dropdown of 50 sampled customers from the dataset, each prefixed with their actual churn outcome. Picking one populates every input field with that customer's real values.
- **Adjust live:** sliders for numeric features (age, balance, credit score, tenure, etc.) and dropdowns for categoricals (location, gender, card type). The model re-scores on every change.
- **See the drivers:** a feature-contribution panel breaks down which inputs are pushing the current prediction toward churn (red) or staying (green), based on the logistic-regression coefficients applied to this customer's scaled features.

## Pipeline

- **Target:** `Exited` (1 = churned, 0 = stayed).
- **Features used:** `CreditScore`, `Location`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCreditCard`, `IsActiveMember`, `EstimatedSalary`, `Satisfaction Score`, `Card Type`, `Point Earned`.
- **Excluded on purpose:** the `Complain` column. It correlates ≈0.99 with `Exited` in this dataset, so including it makes the model trivially accurate but uninteresting: the demo would just become "did this customer complain? yes ⇒ churn." Excluding it forces the model to reason from the actual customer attributes.
- **Pipeline:** `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for categoricals) → `LogisticRegression(max_iter=1000)`. 5-fold stratified cross-validation for honest performance numbers.

## Source code

https://github.com/kspinghar/bank-churn-whatif
