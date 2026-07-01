"""Batch prediction utility for large datasets (efficient chunked processing)."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator, Tuple
import joblib
import logging
from preprocessing import feature_engineering, clean_and_prepare
from path_utils import resolve_path

logger = logging.getLogger(__name__)


class BatchPredictor:
    """Efficiently predict on large datasets using chunked processing."""
    
    def __init__(self, model_path: str = "models/best_model.joblib", chunk_size: int = 5000):
        """
        Initialize batch predictor.
        
        Args:
            model_path: Path to trained model
            chunk_size: Number of rows to process per chunk (5000 works well for ~2GB RAM)
        """
        self.model_path = resolve_path(model_path)
        self.chunk_size = chunk_size
        self.model = None
    
    def load_model(self):
        """Load the trained model (lazy loading)."""
        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        return self.model
    
    def predict_chunks(self, df: pd.DataFrame, return_proba: bool = True) -> Iterator[np.ndarray]:
        """
        Predict on large DataFrame in memory-efficient chunks.
        
        Args:
            df: Input DataFrame
            return_proba: If True, return probabilities; else return class predictions
            
        Yields:
            Prediction arrays for each chunk
        """
        model = self.load_model()
        
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size]
            aligned_chunk = self._align_columns(chunk)
            if return_proba:
                yield model.predict_proba(aligned_chunk)[:, 1]
            else:
                yield model.predict(aligned_chunk)
    
    def predict_on_csv(self, csv_path: str, output_path: str = None, 
                      dtypes: dict = None) -> str:
        """
        Predict on CSV file without loading entire file into memory.
        Results are written directly to output CSV.
        
        Args:
            csv_path: Path to input CSV
            output_path: Path to save predictions (default: same name + _predictions.csv)
            dtypes: Dict of dtypes to use when reading CSV
            
        Returns:
            Path to output CSV file
        """
        model = self.load_model()
        
        if output_path is None:
            csv_path = Path(csv_path)
            output_path = csv_path.parent / f"{csv_path.stem}_predictions.csv"
        
        output_path = Path(output_path)
        
        # Read and process in chunks
        first_chunk = True
        total_rows = 0
        
        for chunk_df in pd.read_csv(csv_path, chunksize=self.chunk_size, dtype=dtypes):
            # Align columns and predict
            chunk_aligned = self._align_columns(chunk_df)
            probs = model.predict_proba(chunk_aligned)[:, 1]
            
            # Create result dataframe
            result_df = chunk_df.copy()
            result_df['churn_probability'] = probs
            result_df['risk_level'] = ['High Risk' if p >= 0.35 else 'Low Risk' for p in probs]
            
            # Write to CSV (append mode after first chunk)
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            result_df.to_csv(output_path, mode=mode, header=header, index=False)
            
            total_rows += len(chunk_df)
            first_chunk = False
            
            logger.info(f"Processed {total_rows} rows... (mean prob: {np.mean(probs):.4f})")
        
        return str(output_path)
    
    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align input DataFrame columns to match trained model by applying feature engineering."""
        model = self.load_model()
        df = df.copy()
        
        logger.debug(f"Input shape: {df.shape}, columns: {df.columns.tolist()}")
        
        # Ensure clean input before feature engineering
        df = clean_and_prepare(df)
        logger.debug(f"After clean_and_prepare: {df.shape}, columns: {df.columns.tolist()}")
        
        df = feature_engineering(df)
        logger.debug(f"After feature_engineering: {df.shape}, columns: {df.columns.tolist()}")
        
        # Get expected columns from preprocessor
        pre = model.named_steps.get("preprocessor", model.named_steps.get("pre"))
        expected_cols = []
        numeric_cols = []
        categorical_cols = []
        
        for name, transformer, cols in pre.transformers_:
            if cols is None or cols == "drop" or cols == "passthrough":
                continue
            cols_list = list(cols)
            expected_cols.extend(cols_list)
            if name == "num":
                numeric_cols.extend(cols_list)
            elif name == "cat":
                categorical_cols.extend(cols_list)
        
        logger.debug(f"Expected numeric cols: {numeric_cols}")
        logger.debug(f"Expected categorical cols: {categorical_cols}")
        
        # Fill missing columns with appropriate defaults
        for col in expected_cols:
            if col not in df.columns:
                if col in numeric_cols:
                    # For missing numeric features, use median (safer than 0)
                    df[col] = np.nan
                    logger.debug(f"Added missing numeric column {col} with NaN")
                else:
                    # For missing categorical, use "unknown"
                    df[col] = "unknown"
                    logger.debug(f"Added missing categorical column {col} with 'unknown'")
        
        # Validate that features are different across rows
        for col in numeric_cols:
            if col in df.columns:
                unique_vals = df[col].nunique()
                logger.debug(f"Numeric col '{col}': {unique_vals} unique values out of {len(df)} rows")
        
        # Select only the columns the model expects, in the correct order
        result = df[expected_cols]
        logger.debug(f"Final aligned shape: {result.shape}")
        
        return result
    
    def get_stats(self, probs: np.ndarray) -> dict:
        """Calculate statistics from prediction array."""
        return {
            'total': len(probs),
            'mean_prob': float(np.mean(probs)),
            'median_prob': float(np.median(probs)),
            'max_prob': float(np.max(probs)),
            'min_prob': float(np.min(probs)),
            'high_risk_count': int(np.sum(probs >= 0.35)),
            'high_risk_pct': float(100 * np.sum(probs >= 0.35) / len(probs))
        }


def process_large_csv(csv_path: str, output_path: str = None, 
                     chunk_size: int = 5000, dtypes: dict = None) -> Tuple[str, dict]:
    """
    Convenience function to process large CSV file.
    
    Args:
        csv_path: Path to input CSV
        output_path: Path to save results
        chunk_size: Rows per chunk
        dtypes: Column dtypes to use
        
    Returns:
        Tuple of (output_path, stats_dict)
    """
    predictor = BatchPredictor(chunk_size=chunk_size)
    output = predictor.predict_on_csv(csv_path, output_path, dtypes)
    
    # Calculate overall stats
    result_df = pd.read_csv(output)
    stats = predictor.get_stats(result_df['churn_probability'].values)
    
    return output, stats
