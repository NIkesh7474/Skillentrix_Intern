"""Generate sample Telco customer churn dataset for testing"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

n_samples = 500

data = {
    'customerID': [f'C{i:05d}' for i in range(n_samples)],
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52]),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    'tenure': np.random.randint(0, 73, n_samples),
    'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.1]),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.45, 0.36, 0.19]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.37, 0.44, 0.19]),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.39, 0.42, 0.19]),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.36, 0.45, 0.19]),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.46, 0.19]),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.39, 0.42, 0.19]),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.43, 0.19]),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41]),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
                                      n_samples, p=[0.33, 0.22, 0.23, 0.22]),
    'MonthlyCharges': np.round(np.random.uniform(18, 118, n_samples), 2),
    'TotalCharges': np.random.uniform(0, 8673, n_samples),
}

# Churn probability depends on contract and tenure
churn_prob = np.where(data['Contract'] == 'Month-to-month', 0.42, 
                      np.where(data['Contract'] == 'One year', 0.11, 0.02))
churn_prob *= (1 - data['tenure'] / 100)  # Less likely to churn with longer tenure
churn_prob = np.clip(churn_prob, 0.01, 0.5)  # Clip to reasonable range

churn_values = (np.random.random(n_samples) < churn_prob).astype(int)
data['Churn'] = ['Yes' if x == 1 else 'No' for x in churn_values]

df = pd.DataFrame(data)

# Create data directory and save
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)
df.to_csv('data/telco_customer_churn.csv', index=False)

print(f"✓ Generated sample dataset with {len(df)} customers")
print(f"✓ Saved to data/telco_customer_churn.csv")
print(f"✓ Churn rate: {sum(1 for x in df['Churn'] if x == 'Yes') / len(df):.2%}")
