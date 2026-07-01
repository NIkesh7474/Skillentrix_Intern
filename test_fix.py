"""
Test script to verify the fix for identical probability issue.
"""
import numpy as np
import pandas as pd
import logging
from preprocessing import clean_and_prepare, feature_engineering, build_preprocessor
from batch_predictor import BatchPredictor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_rows=100):
    """Create test data with varying customer profiles."""
    np.random.seed(42)
    data = {
        'customerID': [f'C{i:05d}' for i in range(n_rows)],
        'gender': np.random.choice(['Male', 'Female'], n_rows),
        'SeniorCitizen': np.random.choice([0, 1], n_rows),
        'Partner': np.random.choice(['Yes', 'No'], n_rows),
        'Dependents': np.random.choice(['Yes', 'No'], n_rows),
        'tenure': np.random.randint(1, 72, n_rows),
        'PhoneService': np.random.choice(['Yes', 'No'], n_rows),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_rows),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_rows),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_rows),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_rows),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_rows),
        'MonthlyCharges': np.random.uniform(20, 120, n_rows),
        'TotalCharges': np.random.uniform(100, 8000, n_rows),
    }
    return pd.DataFrame(data)

def test_feature_variance():
    """Test that features have proper variance."""
    print("\n" + "="*80)
    print("TEST 1: Feature Variance Test")
    print("="*80)
    
    df_test = create_test_data(500)
    print(f"✓ Created test data with {len(df_test)} rows")
    
    # Apply preprocessing
    df_clean = clean_and_prepare(df_test)
    print(f"✓ After clean_and_prepare: {df_clean.shape}")
    
    df_features = feature_engineering(df_clean)
    print(f"✓ After feature_engineering: {df_features.shape}")
    
    # Check feature variance
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    print(f"\nNumeric features: {numeric_cols.tolist()}")
    
    for col in numeric_cols:
        if col != 'Churn':
            unique_vals = df_features[col].nunique()
            std_val = df_features[col].std()
            mean_val = df_features[col].mean()
            print(f"  {col}: unique={unique_vals}, std={std_val:.4f}, mean={mean_val:.4f}")
            
            if unique_vals < 5:
                print(f"    ⚠️  WARNING: Low variance for {col}")
    
    print("\n✓ Feature variance test passed!")
    return True

def test_predictions():
    """Test that predictions vary across different customer profiles."""
    print("\n" + "="*80)
    print("TEST 2: Prediction Variance Test")
    print("="*80)
    
    try:
        predictor = BatchPredictor()
        model = predictor.load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    df_test = create_test_data(200)
    print(f"✓ Created test data with {len(df_test)} rows")
    
    # Prepare data
    df_aligned = predictor._align_columns(df_test)
    print(f"✓ Data aligned: {df_aligned.shape}")
    
    # Make predictions
    probs = model.predict_proba(df_aligned)[:, 1]
    print(f"✓ Predictions made: {len(probs)} probabilities")
    
    # Check variance
    prob_std = np.std(probs)
    prob_mean = np.mean(probs)
    prob_min = np.min(probs)
    prob_max = np.max(probs)
    prob_unique = len(np.unique(probs))
    
    print(f"\nPrediction Statistics:")
    print(f"  Mean: {prob_mean:.4f}")
    print(f"  Std:  {prob_std:.6f}")
    print(f"  Min:  {prob_min:.4f}")
    print(f"  Max:  {prob_max:.4f}")
    print(f"  Unique values: {prob_unique}")
    
    if prob_std < 0.0001:
        print(f"❌ FAILED: Predictions have zero variance! (std={prob_std:.6f})")
        print("   All customers are getting the same probability!")
        return False
    
    if prob_unique < 10:
        print(f"⚠️  WARNING: Low unique values ({prob_unique})")
    
    # Show sample predictions
    print(f"\nSample predictions (first 10):")
    for i in range(min(10, len(probs))):
        print(f"  Customer {i}: {probs[i]:.4f}")
    
    print("\n✓ Prediction variance test passed!")
    return True

def test_large_dataset():
    """Test with larger dataset to check scalability."""
    print("\n" + "="*80)
    print("TEST 3: Large Dataset Test (1000 rows)")
    print("="*80)
    
    try:
        predictor = BatchPredictor(chunk_size=500)
        model = predictor.load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    df_test = create_test_data(1000)
    print(f"✓ Created test data with {len(df_test)} rows")
    
    all_probs = []
    chunk_size = 500
    
    for i in range(0, len(df_test), chunk_size):
        chunk = df_test.iloc[i:i+chunk_size]
        chunk_aligned = predictor._align_columns(chunk)
        chunk_probs = model.predict_proba(chunk_aligned)[:, 1]
        all_probs.extend(chunk_probs)
        print(f"  Chunk {i//chunk_size + 1}: {len(chunk)} rows, prob range [{chunk_probs.min():.4f}, {chunk_probs.max():.4f}]")
    
    all_probs_array = np.array(all_probs)
    print(f"\n✓ Processed all {len(all_probs)} predictions")
    print(f"  Mean: {np.mean(all_probs_array):.4f}")
    print(f"  Std:  {np.std(all_probs_array):.6f}")
    print(f"  Range: [{np.min(all_probs_array):.4f}, {np.max(all_probs_array):.4f}]")
    print(f"  Unique: {len(np.unique(all_probs_array))}")
    
    if np.std(all_probs_array) < 0.0001:
        print(f"❌ FAILED: All predictions identical!")
        return False
    
    print("\n✓ Large dataset test passed!")
    return True

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE FIX VERIFICATION")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Feature Variance", test_feature_variance()))
    results.append(("Prediction Variance", test_predictions()))
    results.append(("Large Dataset", test_large_dataset()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The fix is working correctly.")
        print("   You can now upload a CSV in the Streamlit app.")
    else:
        print("\n❌ Some tests failed. Please review the logs above.")
    
    print("="*80)
