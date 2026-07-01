Customer Churn Prediction

Problem Statement
-----------------
Predict which customers are likely to churn (leave the service). Use predictions to target retention campaigns and estimate ROI.

Dataset
-------
Telco Customer Churn dataset (place CSV at `data/telco_customer_churn.csv`). Typical columns: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`.

Workflow
--------
1. EDA: `notebooks/eda.ipynb` — churn distribution, contract/payment/tenure analysis, correlation heatmap.
2. Preprocessing & feature engineering: numeric imputation, one-hot encoding, tenure bins, `avg_charge_per_month`.
3. Modeling: Logistic Regression, Random Forest, XGBoost with hyperparameter tuning using `RandomizedSearchCV` and `StratifiedKFold` (5-fold).
4. Evaluation: ROC-AUC, precision, recall, F1, confusion matrix, cross-validated mean/std reporting.
5. Explainability: SHAP summary, waterfall, and dependence plots.
6. Deployment: `predict.py` CLI for single-record scoring and Docker support.

How to run
----------
Create venv and install dependencies:

```bash
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

Train and evaluate:

```bash
python churn_prediction.py --data data/telco_customer_churn.csv --output outputs
```

Predict for a single customer JSON:

```bash
python predict.py sample_customer.json
```

Results and outputs
-------------------
- `models/best_model.joblib` — trained pipeline saved
- `outputs/metrics.json` — per-model metrics
- `outputs/roc_curve.png` — ROC curves
- `outputs/feature_importance.png` — top features visualization
- `outputs/confusion_matrix.png` — confusion matrix for best model
- `outputs/shap_summary.png`, `shap_waterfall.png`, `shap_dependence.png` — SHAP explainability visuals

Business insights and next steps
-------------------------------
- Use predicted probabilities to design targeted retention offers; simulate campaign cost vs avoided churn revenue.
- Compute Customer Lifetime Value (CLV) per segment and prioritize high-CLV at-risk customers.
- A/B test retention offers and monitor model drift in production.

Future improvements
-------------------
- Add unit tests under `tests/` for preprocessing and prediction.
- Add a Streamlit dashboard for interactive exploration.
- Build a FastAPI service for REST predictions and deployment.
- Use `Dockerfile` for consistent environment packaging.

Contact
-------
For questions or to request additional features (dashboard, API), open an issue or message the project owner.
