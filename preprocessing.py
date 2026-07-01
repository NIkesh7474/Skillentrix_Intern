import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


FALLBACK_PATTERNS = [
    (("customer", "id"), "customerID"),
    (("cust", "id"), "customerID"),
    (("total", "charg"), "TotalCharges"),
    (("monthly", "charg"), "MonthlyCharges"),
    (("month", "charg"), "MonthlyCharges"),
    (("tenure",), "tenure"),
    (("senior", "citizen"), "SeniorCitizen"),
    (("gender",), "gender"),
    (("partner",), "Partner"),
    (("depend",), "Dependents"),
    (("phone", "service"), "PhoneService"),
    (("multiple", "line"), "MultipleLines"),
    (("internet", "service"), "InternetService"),
    (("online", "security"), "OnlineSecurity"),
    (("online", "backup"), "OnlineBackup"),
    (("device", "protection"), "DeviceProtection"),
    (("tech", "support"), "TechSupport"),
    (("streaming", "tv"), "StreamingTV"),
    (("streaming", "movie"), "StreamingMovies"),
    (("contract",), "Contract"),
    (("paperless", "billing"), "PaperlessBilling"),
    (("payment",), "PaymentMethod"),
    (("churn",), "Churn"),
]


def normalize_column_name(column_name: str) -> str:
    """Normalize common column name variations to canonical training names."""
    original = str(column_name).strip()
    normalized = re.sub(r"[^0-9a-z]+", "", original.lower())
    aliases = {
        "customerid": "customerID",
        "custid": "customerID",
        "totalcharges": "TotalCharges",
        "totalcharge": "TotalCharges",
        "monthlycharges": "MonthlyCharges",
        "monthlycharge": "MonthlyCharges",
        "tenure": "tenure",
        "gender": "gender",
        "partner": "Partner",
        "dependents": "Dependents",
        "phoneservice": "PhoneService",
        "multiplelines": "MultipleLines",
        "internetservice": "InternetService",
        "onlinesecurity": "OnlineSecurity",
        "onlinebackup": "OnlineBackup",
        "deviceprotection": "DeviceProtection",
        "techsupport": "TechSupport",
        "streamingtv": "StreamingTV",
        "streamingmovies": "StreamingMovies",
        "contract": "Contract",
        "paperlessbilling": "PaperlessBilling",
        "paymentmethod": "PaymentMethod",
        "seniorcitizen": "SeniorCitizen",
        "churn": "Churn",
    }
    if normalized in aliases:
        return aliases[normalized]
    for tokens, canonical in FALLBACK_PATTERNS:
        if all(token in normalized for token in tokens):
            return canonical
    return original


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    logger.debug(f"Normalized columns: {df.columns.tolist()}")
    return df


def parse_numeric_series(series: pd.Series) -> pd.Series:
    """Clean numeric strings with commas, currency symbols, and whitespace."""
    if series.dtype == object or pd.api.types.is_string_dtype(series.dtype):
        cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        cleaned = cleaned.replace("", np.nan)
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data with robust handling of missing columns."""
    df = df.copy()
    df = normalize_columns(df)
    logger.debug(f"Columns after normalization: {df.columns.tolist()}")

    # Drop customer ID columns if present
    for col in ("customerID", "customer_id", "CustomerID"):
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.debug(f"Dropped column: {col}")
            break

    # Ensure required numeric columns exist and are properly typed
    required_numeric = ["TotalCharges", "MonthlyCharges", "tenure", "SeniorCitizen"]
    for col in required_numeric:
        if col in df.columns:
            df[col] = parse_numeric_series(df[col])
            logger.debug(f"Parsed numeric column {col}: min={df[col].min()}, max={df[col].max()}, null_count={df[col].isna().sum()}")
        else:
            # If column is missing, create it with default values (will be imputed later)
            if col == "tenure":
                df[col] = 12.0  # Default tenure
            elif col == "SeniorCitizen":
                df[col] = 0  # Default non-senior
            elif col == "MonthlyCharges":
                df[col] = 50.0  # Default monthly charge
            elif col == "TotalCharges":
                df[col] = 600.0  # Default total charges
            logger.debug(f"Missing column {col} - created with default value")

    # Handle SeniorCitizen specifically
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = parse_numeric_series(df["SeniorCitizen"]).fillna(0).astype(int)

    # Map Churn column if present
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(df["Churn"])

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized feature engineering for memory efficiency on large datasets."""
    df = df.copy()
    df = normalize_columns(df)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    # Use alternate feature fallbacks when some columns are missing
    if "TotalCharges" not in df.columns and "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).replace([np.inf, -np.inf], np.nan)
        logger.debug("Computed TotalCharges from MonthlyCharges and tenure")
    if "MonthlyCharges" not in df.columns and "TotalCharges" in df.columns and "tenure" in df.columns:
        df["MonthlyCharges"] = (df["TotalCharges"] / df["tenure"]).replace([np.inf, -np.inf], np.nan)
        logger.debug("Computed MonthlyCharges from TotalCharges and tenure")
    if "tenure" not in df.columns and "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        df["tenure"] = (df["TotalCharges"] / df["MonthlyCharges"]).replace([np.inf, -np.inf], np.nan)
        logger.debug("Estimated tenure from TotalCharges and MonthlyCharges")
    if "tenure" not in df.columns and "MonthlyCharges" in df.columns and "TotalCharges" not in df.columns:
        df["tenure"] = 12.0
        logger.debug("Filled missing tenure with default 12")
    if "TotalCharges" not in df.columns and "tenure" in df.columns and "MonthlyCharges" not in df.columns:
        df["TotalCharges"] = 600.0
        logger.debug("Filled missing TotalCharges with default 600")
    if "MonthlyCharges" not in df.columns and "tenure" in df.columns and "TotalCharges" not in df.columns:
        df["MonthlyCharges"] = 50.0
        logger.debug("Filled missing MonthlyCharges with default 50")

    if "tenure" in df.columns:
        df["tenure_bin"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 60, 1000],
            labels=["0-12", "13-24", "25-48", "49-60", "61+"],
        )
        logger.debug(f"Tenure_bin distribution: {df['tenure_bin'].value_counts().to_dict()}")

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        safe_tenure = df["tenure"].replace(0, np.nan)
        df["avg_charge_per_month"] = (df["TotalCharges"] / safe_tenure)
        df["avg_charge"] = df["avg_charge_per_month"].copy()
        if "MonthlyCharges" in df.columns:
            fill_mask = df["avg_charge"].isna() | (df["avg_charge"] == 0)
            df.loc[fill_mask, "avg_charge"] = df.loc[fill_mask, "MonthlyCharges"]
            df.loc[fill_mask, "avg_charge_per_month"] = df.loc[fill_mask, "MonthlyCharges"]
        logger.debug(f"avg_charge_per_month: min={df['avg_charge_per_month'].min()}, max={df['avg_charge_per_month'].max()}, mean={df['avg_charge_per_month'].mean()}")
    elif "MonthlyCharges" in df.columns:
        df["avg_charge_per_month"] = df["MonthlyCharges"].copy()
        df["avg_charge"] = df["MonthlyCharges"].copy()
        logger.debug("Set avg_charge_per_month and avg_charge from MonthlyCharges only")

    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        tenure_safe = df["tenure"].clip(lower=1)
        df["clv"] = (df["MonthlyCharges"] * tenure_safe)
        logger.debug(f"CLV: min={df['clv'].min()}, max={df['clv'].max()}, mean={df['clv'].mean()}")
    elif "MonthlyCharges" in df.columns:
        df["clv"] = df["MonthlyCharges"].copy()
        logger.debug("Set clv from MonthlyCharges only")

    for col in ["avg_charge_per_month", "clv", "tenure_bin"]:
        if col in df.columns:
            unique_count = df[col].nunique(dropna=True)
            logger.debug(f"Feature {col} has {unique_count} unique values in {len(df)} rows")

    return df


def build_preprocessor(df: pd.DataFrame):
    """Build preprocessor with robust column handling."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Churn"]
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "Churn"]

    # Ensure all required feature columns are present
    required_features = ["tenure_bin", "avg_charge_per_month", "clv"]
    for feat in required_features:
        if feat not in numeric_cols + categorical_cols:
            if feat == "tenure_bin":
                categorical_cols.append(feat)
                logger.debug(f"Added {feat} to categorical columns")
            else:
                numeric_cols.append(feat)
                logger.debug(f"Added {feat} to numeric columns")

    logger.debug(f"Numeric columns: {numeric_cols}")
    logger.debug(f"Categorical columns: {categorical_cols}")

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])
    return preprocessor, numeric_cols, categorical_cols
