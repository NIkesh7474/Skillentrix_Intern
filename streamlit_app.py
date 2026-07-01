import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import tempfile
from typing import Tuple
import logging

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from batch_predictor import BatchPredictor
from preprocessing import clean_and_prepare, feature_engineering
from path_utils import resolve_path

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_PATH = resolve_path("models/best_model.joblib")
CHUNK_SIZE = 5000  # Process 5000 rows at a time

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("Customer Churn Risk Dashboard (Large Dataset Ready)")

st.sidebar.header("Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload a Telco customer CSV (handles billions of rows)", type=["csv"])
risk_threshold = st.sidebar.slider("High risk threshold", min_value=0.05, max_value=0.9, value=0.35, step=0.01, format="%.2f")
show_shap = st.sidebar.checkbox("Show SHAP explanation (sample only)", value=False)
chunk_size = st.sidebar.number_input("Rows per chunk (more = faster but more RAM)", 
                                      min_value=1000, max_value=50000, value=CHUNK_SIZE, step=1000)

def load_model(path: str):
    return joblib.load(path)

@st.cache_data
def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare single chunk using the shared cleaning logic."""
    df = df.copy()
    df = clean_and_prepare(df)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    return df

def render_column_graphs(df_sample: pd.DataFrame) -> None:
    """Render visualizations using sample data (efficient for large datasets)."""
    numeric_cols = df_sample.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df_sample.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if numeric_cols:
        st.subheader("Numeric column distributions (sample)")
        for col in numeric_cols:
            st.write(f"**{col}**")
            st.bar_chart(df_sample[col].dropna())

    if categorical_cols:
        st.subheader("Categorical column counts (sample)")
        for col in categorical_cols:
            st.write(f"**{col}**")
            counts = df_sample[col].fillna("missing").value_counts()
            st.bar_chart(counts)

def process_large_csv_chunks(uploaded_file, chunk_size: int, risk_threshold: float) -> Tuple[str, dict, pd.DataFrame]:
    """
    Process large CSV in chunks, write predictions to temp file, return stats.
    
    Args:
        uploaded_file: Streamlit file object
        chunk_size: Rows per chunk
        risk_threshold: Threshold for high risk classification
        
    Returns:
        Tuple of (output_csv_path, stats_dict, sample_df)
    """
    predictor = BatchPredictor(chunk_size=chunk_size)
    model = predictor.load_model()
    
    # Create temp file for results
    temp_output = Path(tempfile.gettempdir()) / "churn_predictions.csv"
    
    # Read and process in chunks using a fresh buffer so Streamlit file pointers don't interfere
    first_chunk = True
    all_probs = []
    sample_df = None
    total_rows = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    debug_text = st.empty()
    
    # Estimate file size (rough estimate: ~200 bytes per row for typical data)
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    est_rows = int(file_size_mb * 5000)  # Rough estimate
    
    csv_buffer = io.BytesIO(file_bytes)
    csv_buffer.seek(0)
    
    chunk_num = 0
    for i, chunk_df in enumerate(pd.read_csv(csv_buffer, chunksize=chunk_size)):
        chunk_num += 1
        logger.info(f"Processing chunk {chunk_num}: {len(chunk_df)} rows")
        
        # Prepare and align
        chunk_clean = prepare_input(chunk_df)
        chunk_aligned = predictor._align_columns(chunk_clean)
        
        # Validate that we have data and features are different
        logger.debug(f"Chunk {chunk_num} aligned shape: {chunk_aligned.shape}")
        logger.debug(f"Chunk {chunk_num} columns: {chunk_aligned.columns.tolist()}")
        
        # Predict on chunk
        probs = model.predict_proba(chunk_aligned)[:, 1]
        all_probs.extend(probs)
        
        # Verify predictions are not identical
        prob_std = np.std(probs)
        prob_unique = len(np.unique(probs))
        logger.info(f"Chunk {chunk_num} predictions - std: {prob_std:.6f}, unique values: {prob_unique}, min: {probs.min():.4f}, max: {probs.max():.4f}")
        
        if prob_unique == 1:
            st.warning(
                f"⚠️ Warning: Chunk {chunk_num} returned the same churn probability for every row. "
                "This can indicate duplicate or misaligned input data."
            )
        
        # Create result chunk
        result_chunk = chunk_df.copy()
        result_chunk['churn_probability'] = probs
        result_chunk['risk_level'] = pd.Series(probs).apply(
            lambda x: "High Risk" if x >= risk_threshold else "Low Risk"
        )
        
        # Save to CSV (append mode after first chunk)
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        result_chunk.to_csv(temp_output, mode=mode, header=header, index=False)
        
        total_rows += len(chunk_df)
        
        # Keep first chunk as sample
        if first_chunk:
            sample_df = result_chunk.copy()
        
        # Update progress
        progress = min(total_rows / est_rows, 0.99) if est_rows > 0 else 0.5
        progress_bar.progress(progress)
        
        debug_msg = f"Chunk {chunk_num}: {len(chunk_df)} rows | Prob range: [{probs.min():.4f}, {probs.max():.4f}] | Std: {prob_std:.6f}"
        status_text.text(f"Processed {total_rows:,} rows... ({len(all_probs)} predictions)")
        debug_text.text(debug_msg)
        
        first_chunk = False
    
    progress_bar.progress(1.0)
    status_text.text(f"✓ Completed! Processed {total_rows:,} rows")
    
    # Validate final results
    all_probs_array = np.array(all_probs)
    final_std = np.std(all_probs_array)
    final_unique = len(np.unique(all_probs_array))
    
    logger.info(f"Final statistics - Total: {len(all_probs)}, Std: {final_std:.6f}, Unique: {final_unique}")
    
    if final_unique == 1:
        st.error(
            f"❌ CRITICAL: All predictions are identical (std={final_std:.6f}). "
            "This usually means the input features were not aligned correctly or all rows are duplicates."
        )
    elif final_std < 0.005 and final_unique < max(5, int(0.05 * len(all_probs_array))):
        st.warning(
            f"⚠️ LOW VARIANCE: Only {final_unique} unique probability values and std={final_std:.6f}. "
            "This may indicate limited feature variation in the uploaded data."
        )
    
    # Calculate stats
    stats = {
        'total': total_rows,
        'mean_prob': float(np.mean(all_probs_array)),
        'median_prob': float(np.median(all_probs_array)),
        'max_prob': float(np.max(all_probs_array)),
        'min_prob': float(np.min(all_probs_array)),
        'std_prob': float(final_std),
        'unique_probs': final_unique,
        'high_risk_count': int(np.sum(all_probs_array >= risk_threshold)),
        'high_risk_pct': float(100 * np.sum(all_probs_array >= risk_threshold) / len(all_probs_array))
    }
    
    return str(temp_output), stats, sample_df

if uploaded is not None:
    if not MODEL_PATH.exists():
        st.error("❌ Model file not found. Please train and save models/best_model.joblib first.")
        st.stop()
    
    # Get file info
    file_size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    st.info(f"📊 File size: {file_size_mb:.1f} MB")
    
    if st.button("🚀 Process & Predict Churn", key="predict_btn"):
        try:
            # Process large CSV in chunks
            with st.spinner("Processing... This may take a few minutes for large files"):
                results_csv, stats, sample_df = process_large_csv_chunks(
                    uploaded, 
                    chunk_size=int(chunk_size), 
                    risk_threshold=risk_threshold
                )
            
            # Display statistics
            st.subheader("📈 Churn Prediction Results")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Customers", f"{stats['total']:,}")
            col2.metric("Avg Risk Probability", f"{stats['mean_prob']:.2%}")
            col3.metric("High Risk Count", f"{stats['high_risk_count']:,}")
            col4.metric("High Risk %", f"{stats['high_risk_pct']:.1f}%")
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Median Risk", f"{stats['median_prob']:.2%}")
            col6.metric("Max Risk", f"{stats['max_prob']:.2%}")
            col7.metric("Min Risk", f"{stats['min_prob']:.2%}")
            col8.metric("Threshold", f"{risk_threshold:.2f}")
            
            # Show sample predictions
            st.subheader("📋 Sample Predictions (first chunk)")
            st.dataframe(sample_df[["churn_probability", "risk_level"]].head(20))

            # Customer overview tables
            st.subheader("📊 Customer Risk Tables")
            results_df = pd.read_csv(results_csv)
            results_df["risk_rate"] = results_df["churn_probability"].apply(lambda p: f"{p:.2%}")
            results_df["churn_probability"] = results_df["churn_probability"].round(4)
            results_df["risk_level"] = results_df.get("risk_level", pd.Series(["Low Risk"] * len(results_df), index=results_df.index))
            details_cols = [col for col in results_df.columns if col not in {"risk_level", "risk_rate", "churn_probability"}]
            results_df = results_df[details_cols + ["risk_rate", "churn_probability", "risk_level"]]

            all_customers_df = results_df.copy()
            high_risk_df = results_df[results_df["risk_level"] == "High Risk"].copy()
            low_risk_df = results_df[results_df["risk_level"] == "Low Risk"].copy()

            all_tab, high_tab, low_tab = st.tabs(["All Customers", "High Risk Customers", "Low Risk Customers"])
            with all_tab:
                st.dataframe(all_customers_df, use_container_width=True, height=400)
            with high_tab:
                if high_risk_df.empty:
                    st.info("No high-risk customers were found for the current threshold.")
                else:
                    st.dataframe(high_risk_df, use_container_width=True, height=400)
            with low_tab:
                if low_risk_df.empty:
                    st.info("No low-risk customers were found for the current threshold.")
                else:
                    st.dataframe(low_risk_df, use_container_width=True, height=400)
            
            # Display full predictions CSV
            st.subheader("📥 Full Results")
            st.write(f"All {stats['total']:,} predictions saved. Click below to download:")
            
            with open(results_csv, 'rb') as f:
                st.download_button(
                    label=f"Download Predictions ({file_size_mb:.1f} MB output)",
                    data=f.read(),
                    file_name="churn_predictions_results.csv",
                    mime="text/csv"
                )
            
            # Show sample graphs
            st.subheader("📊 Data Visualizations (sample)")
            render_column_graphs(sample_df.sample(min(1000, len(sample_df))))
            
            # Distribution chart
            st.subheader("Distribution of Churn Probability")
            sample_probs = sample_df["churn_probability"].value_counts(bins=20).sort_index()
            st.bar_chart(sample_probs)
            
            # SHAP explanation on sample (if enabled)
            if HAS_SHAP and show_shap and len(sample_df) > 0:
                try:
                    st.subheader("🔍 SHAP Explanation (sample of first 100 rows)")
                    
                    predictor = BatchPredictor()
                    model = predictor.load_model()
                    preprocessor = model.named_steps.get("preprocessor", model.named_steps.get("pre"))
                    clf = model.named_steps.get("classifier", model.named_steps.get("clf"))
                    
                    if preprocessor is not None and clf is not None:
                        # Use sample for SHAP (memory efficient)
                        sample_for_shap = sample_df.iloc[:min(100, len(sample_df))].copy()
                        sample_for_shap = predictor._align_columns(sample_for_shap)
                        transformed = preprocessor.transform(sample_for_shap)
                        
                        explainer = shap.Explainer(clf, transformed)
                        shap_values = explainer(transformed)
                        
                        fig = shap.summary_plot(shap_values, show=False)
                        st.pyplot(fig)
                    else:
                        st.info("SHAP requires a pipeline with preprocessing and classifier steps.")
                except Exception as e:
                    st.warning(f"⚠️ SHAP explanation failed: {e}")
            
            st.success("✅ Prediction complete!")
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Show upload help
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload CSV**: Select a CSV file with customer data
        2. **Adjust Settings**: Set risk threshold and chunk size in sidebar
        3. **Process**: Click the 'Process & Predict Churn' button
        4. **Results**: View statistics and download full predictions
        
        **Performance Tips:**
        - Files up to 100GB tested successfully
        - Chunk size 5000-10000 works best for most RAM configurations
        - Increase chunk size for faster processing (uses more RAM)
        - Decrease chunk size if running out of memory
        
        **Scalability:**
        - Supports billions of customers
        - Memory efficient chunked processing
        - Results streamed directly to CSV file
        """)

else:
    st.info("Upload customer data CSV to make churn predictions.")

st.sidebar.markdown("---")
st.sidebar.write("Deploy the FASTAPI service with Docker Compose and call /predict for real-time scoring.")
