"""Simple prediction CLI: python predict.py customer.json
The JSON should contain a flat mapping of feature names to values (matching training features).

For large CSV files, use:
    python predict.py --csv input.csv --output output.csv --chunk-size 5000
"""
import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from batch_predictor import BatchPredictor
from preprocessing import feature_engineering, clean_and_prepare
from path_utils import resolve_path

MODEL_PATH = resolve_path("models/best_model.joblib")


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare input data - clean and remove ID columns. Feature engineering happens in _align_columns."""
    df = df.copy()
    # Remove ID columns
    for col in ("customerID", "customer_id", "CustomerID"):
        if col in df.columns:
            df = df.drop(columns=[col])
            break
    # Remove Churn column if present (it's the target, not a feature)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def load_customer(path):
    """Load single customer from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame([data])


def predict_json_file(json_path: str) -> float:
    """Predict for a single customer from JSON file."""
    cust_path = Path(json_path)
    if not cust_path.exists():
        raise FileNotFoundError(f"File not found: {cust_path}")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    df = load_customer(cust_path)
    df = prepare_input(df)
    
    predictor = BatchPredictor()
    model = predictor.load_model()
    df_aligned = predictor._align_columns(df)
    
    proba = model.predict_proba(df_aligned)[:, 1][0]
    return proba


def predict_csv_file(csv_path: str, output_path: str = None, chunk_size: int = 5000) -> str:
    """Predict on entire CSV file using batch processing."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_predictions.csv"
    
    print(f"Processing {csv_path.name}...")
    predictor = BatchPredictor(chunk_size=chunk_size)
    
    model = predictor.load_model()
    output_path = Path(output_path)
    
    # Read and process in chunks
    first_chunk = True
    total_rows = 0
    
    for chunk_df in pd.read_csv(csv_path, chunksize=chunk_size):
        # Prepare and align
        chunk_clean = prepare_input(chunk_df)
        chunk_aligned = predictor._align_columns(chunk_clean)
        
        # Predict on chunk
        probs = model.predict_proba(chunk_aligned)[:, 1]
        
        # Create result chunk
        result_chunk = chunk_df.copy()
        result_chunk['churn_probability'] = probs
        result_chunk['risk_level'] = pd.Series(probs).apply(
            lambda x: "High Risk" if x >= 0.35 else "Low Risk"
        )
        
        # Save to CSV (append mode after first chunk)
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        result_chunk.to_csv(output_path, mode=mode, header=header, index=False)
        
        total_rows += len(chunk_df)
        print(f"  Processed {total_rows:,} rows...")
        first_chunk = False
    
    print(f"✓ Predictions saved to {output_path}")
    return str(output_path)


def main():
    """Main CLI entry point supporting both JSON and CSV."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Churn prediction CLI")
    parser.add_argument("input", nargs="?", help="Input file (JSON or CSV)")
    parser.add_argument("--csv", help="Process as CSV file")
    parser.add_argument("--json", help="Process as JSON file")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for CSV processing")
    
    args = parser.parse_args()
    
    # Determine input type
    if args.csv:
        csv_file = args.csv
    elif args.json:
        json_file = args.json
    elif args.input:
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".csv":
            csv_file = args.input
        elif input_path.suffix.lower() == ".json":
            json_file = args.input
        else:
            print("Error: File must be .csv or .json")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
    
    # Process based on type
    try:
        if 'csv_file' in locals():
            # CSV batch processing
            result = predict_csv_file(csv_file, args.output, args.chunk_size)
            print(f"\n✓ Results saved to: {result}")
        elif 'json_file' in locals():
            # Single JSON prediction
            proba = predict_json_file(json_file)
            risk_level = "High Risk" if proba >= 0.35 else "Low Risk"
            print(f"\n✓ Churn Probability: {proba*100:.2f}%")
            print(f"✓ Risk Level: {risk_level}")
        else:
            print("Error: No input file specified")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
