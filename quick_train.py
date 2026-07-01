"""Train a churn prediction model using the requested minimal feature set."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_score, recall_score

RAW_DATA_PATH = Path('data/telco_customer_churn.csv')
MODEL_PATH = Path('models/best_model.joblib')


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if 'tenure' in df.columns:
        df['tenure_bin'] = pd.cut(
            df['tenure'],
            bins=[-1, 12, 24, 48, 60, 1000],
            labels=['0-12', '13-24', '25-48', '49-60', '61+'],
        )
    if set(['TotalCharges', 'tenure']).issubset(df.columns):
        df['avg_charge'] = df.apply(
            lambda r: r['TotalCharges'] / r['tenure'] if r['tenure'] and r['tenure'] > 0 else r['MonthlyCharges'],
            axis=1,
        )
    return df


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Churn' not in df.columns:
        raise ValueError('Training data must contain Churn column')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)
    df = df.drop(columns=['customerID']) if 'customerID' in df.columns else df
    df = df.dropna(subset=['Churn'])
    return df


def main():
    df = load_data(RAW_DATA_PATH)
    df = prepare_features(df)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, categorical_cols),
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1))
    ])

    print('Training model using only the requested feature set...')
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f'✓ ROC-AUC: {auc:.4f}')
    print(f'✓ Precision: {prec:.4f}')
    print(f'✓ Recall: {rec:.4f}')

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f'✓ Trained model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
