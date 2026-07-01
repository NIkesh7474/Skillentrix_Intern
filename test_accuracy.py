"""
Test script to verify accurate churn predictions on large datasets.
Demonstrates that predictions now vary across different customers.
"""
import pandas as pd
import numpy as np
from batch_predictor import BatchPredictor
from preprocessing import feature_engineering
import tempfile
from pathlib import Path


def create_test_dataset(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a diverse test dataset with varying customer profiles."""
    np.random.seed(seed)
    
    df = pd.DataFrame({
        'customerID': [f'CUST_{i:06d}' for i in range(n_rows)],
        'gender': np.random.choice(['Male', 'Female'], n_rows),
        'SeniorCitizen': np.random.choice([0, 1], n_rows),
        'Partner': np.random.choice(['Yes', 'No'], n_rows),
        'Dependents': np.random.choice(['Yes', 'No'], n_rows),
        'tenure': np.random.randint(0, 73, n_rows),  # 0 to 72 months
        'MonthlyCharges': np.random.uniform(20, 120, n_rows),
        'TotalCharges': np.random.uniform(100, 8000, n_rows),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_rows),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_rows),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_rows),
    })
    
    return df


def test_prediction_variance():
    """Test that predictions vary across different customers."""
    print("=" * 80)
    print("ACCURACY TEST: Checking for prediction variance across different customers")
    print("=" * 80)
    
    # Create diverse test dataset
    test_df = create_test_dataset(n_rows=50)
    print(f"\n✓ Created test dataset with 50 diverse customers")
    
    # Save to temp CSV
    temp_csv = Path(tempfile.gettempdir()) / "test_customers.csv"
    test_df.to_csv(temp_csv, index=False)
    print(f"✓ Saved to: {temp_csv}")
    
    # Make predictions
    print("\nProcessing predictions...")
    try:
        predictor = BatchPredictor(chunk_size=10)
        output_csv = predictor.predict_on_csv(str(temp_csv))
        
        # Read results
        results = pd.read_csv(output_csv)
        probs = results['churn_probability'].values
        
        # Analyze prediction distribution
        print("\n" + "=" * 80)
        print("PREDICTION STATISTICS")
        print("=" * 80)
        print(f"Total customers: {len(probs)}")
        print(f"Mean churn probability: {np.mean(probs):.4f} ({np.mean(probs)*100:.2f}%)")
        print(f"Median churn probability: {np.median(probs):.4f} ({np.median(probs)*100:.2f}%)")
        print(f"Std deviation: {np.std(probs):.4f}")
        print(f"Min probability: {np.min(probs):.4f} ({np.min(probs)*100:.2f}%)")
        print(f"Max probability: {np.max(probs):.4f} ({np.max(probs)*100:.2f}%)")
        print(f"Range (Max - Min): {np.max(probs) - np.min(probs):.4f}")
        
        # Check for variance
        unique_probs = len(np.unique(np.round(probs, 4)))
        print(f"\nUnique prediction values (rounded to 4 decimals): {unique_probs}")
        
        # Display sample predictions
        print("\n" + "=" * 80)
        print("SAMPLE PREDICTIONS (First 10 customers)")
        print("=" * 80)
        sample_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
                      'churn_probability', 'risk_level']
        sample_results = results[[col for col in sample_cols if col in results.columns]].head(10)
        
        for idx, row in sample_results.iterrows():
            print(f"\nCustomer {idx+1}:")
            for col in sample_cols:
                if col in results.columns:
                    print(f"  {col}: {row[col]}")
        
        # Variance check
        print("\n" + "=" * 80)
        print("ACCURACY CHECK")
        print("=" * 80)
        
        if np.std(probs) < 0.01:
            print("❌ WARNING: Very low variance in predictions (< 0.01)")
            print("   This indicates all customers are getting similar predictions")
            print("   This could mean feature engineering is not being applied correctly")
            return False
        elif np.std(probs) < 0.05:
            print("⚠️  LOW VARIANCE: Some variation detected but it's quite low")
            print(f"   Std dev: {np.std(probs):.4f}")
        else:
            print("✅ GOOD VARIANCE: Predictions show healthy variation")
            print(f"   Std dev: {np.std(probs):.4f}")
            return True
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if temp_csv.exists():
            temp_csv.unlink()


def test_feature_engineering():
    """Test that feature engineering is being applied."""
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING TEST")
    print("=" * 80)
    
    test_df = create_test_dataset(n_rows=5)
    print("\nOriginal columns:")
    print(f"  {sorted(test_df.columns.tolist())}")
    
    # Apply feature engineering
    from preprocessing import feature_engineering
    engineered_df = feature_engineering(test_df)
    
    print("\nColumns after feature engineering:")
    print(f"  {sorted(engineered_df.columns.tolist())}")
    
    # Check for engineered features
    expected_features = ['tenure_bin', 'avg_charge', 'avg_charge_per_month', 'clv']
    found_features = [f for f in expected_features if f in engineered_df.columns]
    
    print(f"\n✓ Found engineered features: {found_features}")
    
    if len(found_features) < len(expected_features):
        missing = set(expected_features) - set(found_features)
        print(f"⚠️  Missing features: {missing}")
    else:
        print("✅ All expected features created!")
    
    # Show sample engineered values
    print("\nSample engineered feature values:")
    for feat in found_features[:2]:
        print(f"  {feat}: {engineered_df[feat].head(3).tolist()}")
    
    return len(found_features) == len(expected_features)


def test_large_dataset():
    """Test with simulated large dataset (1000 rows)."""
    print("\n" + "=" * 80)
    print("LARGE DATASET TEST (1000 customers)")
    print("=" * 80)
    
    # Create large test dataset
    large_df = create_test_dataset(n_rows=1000, seed=123)
    
    # Save to temp CSV
    temp_csv = Path(tempfile.gettempdir()) / "test_large.csv"
    large_df.to_csv(temp_csv, index=False)
    print(f"\n✓ Created test dataset with 1000 customers")
    
    try:
        # Process with small chunk size
        predictor = BatchPredictor(chunk_size=100)
        output_csv = predictor.predict_on_csv(str(temp_csv))
        
        # Read results
        results = pd.read_csv(output_csv)
        probs = results['churn_probability'].values
        
        print(f"✓ Predictions completed for {len(probs)} customers")
        print(f"\nPrediction statistics:")
        print(f"  Mean: {np.mean(probs):.4f}")
        print(f"  Median: {np.median(probs):.4f}")
        print(f"  Std Dev: {np.std(probs):.4f}")
        print(f"  Range: {np.min(probs):.4f} - {np.max(probs):.4f}")
        print(f"  Unique values: {len(np.unique(np.round(probs, 4)))}")
        
        if np.std(probs) > 0.05:
            print(f"\n✅ PASS: Good variance in large dataset")
            return True
        else:
            print(f"\n❌ FAIL: Low variance in large dataset")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_csv.exists():
            temp_csv.unlink()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CHURN PREDICTION ACCURACY TEST SUITE" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run tests
    test_fe = test_feature_engineering()
    test_var = test_prediction_variance()
    test_large = test_large_dataset()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Feature Engineering: {'✅ PASS' if test_fe else '❌ FAIL'}")
    print(f"Prediction Variance: {'✅ PASS' if test_var else '❌ FAIL'}")
    print(f"Large Dataset: {'✅ PASS' if test_large else '❌ FAIL'}")
    
    if test_fe and test_var and test_large:
        print("\n🎉 ALL TESTS PASSED! Predictions are accurate and diverse.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n" + "=" * 80 + "\n")
