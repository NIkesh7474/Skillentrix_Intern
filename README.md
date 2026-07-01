Customer Churn Prediction
----

Customer attrition, also known as customer churn, customer turnover, or customer defection, is the loss of clients or customers.

Telephone service companies, Internet service providers, pay TV companies, insurance firms, and alarm monitoring services, often use customer attrition analysis and customer attrition rates as one of their key business metrics because the cost of retaining an existing customer is far less than acquiring a new one. Companies from these sectors often have customer service branches which attempt to win back defecting clients, because recovered long-term customers can be worth much more to a company than newly recruited clients.

Predictive analytics use churn prediction models that predict customer churn by assessing their propensity of risk to churn. Since these models generate a small prioritized list of potential defectors, they are effective at focusing customer retention marketing programs on the subset of the customer base who are most vulnerable to churn.

In this project I aim to perform customer survival analysis and build a model which can predict customer churn. I also aim to build an app which can be used to understand why a specific customer would stop the service and to know his/her expected lifetime value.

Customer Churn Prediction Dashboard
-----

<img width="1913" height="852" alt="Screenshot 2026-07-01 174415" src="https://github.com/user-attachments/assets/bedc841b-d7f9-4751-bbab-5eaac89d4451" />


<img width="1917" height="857" alt="Screenshot 2026-07-01 175145" src="https://github.com/user-attachments/assets/51f45739-18db-49cb-ab62-6c8b78c8c246" />


<img width="1901" height="858" alt="Screenshot 2026-07-01 175213" src="https://github.com/user-attachments/assets/87fee9ac-dab2-401f-b3c2-0147c995afa6" />


<img width="1902" height="837" alt="Screenshot 2026-07-01 175226" src="https://github.com/user-attachments/assets/ddb15a47-6f6f-4987-b0f1-61b65f3153d9" />


<img width="1888" height="856" alt="Screenshot 2026-07-01 175240" src="https://github.com/user-attachments/assets/8f884a56-8170-481e-b68c-993e0fb39fcd" />


<img width="1908" height="863" alt="Screenshot 2026-07-01 175257" src="https://github.com/user-attachments/assets/e44e978a-3c8b-4621-9f48-3f46288dcf99" />


<img width="1901" height="851" alt="Screenshot 2026-07-01 175320" src="https://github.com/user-attachments/assets/703b4105-88eb-45d0-94aa-3c688e91a00f" />


Project Structure
---
Customer Churn Prediction/

│

├── .github/

│   └── workflows/

│       └── test.yml                  # GitHub Actions CI

│

├── api/

│   └── main.py                       # FastAPI backend

│

├── data/

│   └── telco_customer_churn.csv      # Dataset

│

├── models/

│   └── best_model.joblib             # Saved trained model

│

├── notebooks/

│   └── eda.ipynb                     # Exploratory Data Analysis

│

├── outputs/

│   ├── business_impact.json

│   ├── business_insights.md

│   ├── confusion_matrix.png

│   ├── feature_importance.png

│   ├── metrics.json

│   ├── roc_curve.png

│   ├── shap_summary.png

│   └── test_preprocess/

│       └── prepared.csv

│

├── tests/

│   ├── test_pipeline.py

│   └── test_predict.py

│

├── ACCURACY_FIX.md

├── batch_predictor.py

├── churn_prediction.py

├── config.py

├── create_sample_data.py

├── deploy.py

├── docker-compose.yml

├── Dockerfile

├── evaluate.py

├── FIX_SUMMARY.md

├── generate_sample_data.py

├── mlflow_utils.py

├── monitoring.py

├── OPTIMIZATION_SUMMARY.md

├── params.yaml

├── predict.py

├── preprocessing.py

├── quick_train.py

├── README.md

├── requirements.txt

├── retrain.py

├── sample_customers_1000.csv

├── SCALABILITY_GUIDE.md

├── streamlit_app.py

├── temp_predictions.csv

├── test_accuracy.py

├── test_fix.py

├── train.py

└── uplift.py


Things That I added in the project
----
1. Customer Churn Prediction

The project builds a classification model to predict whether a customer is likely to leave a telecom company. It uses customer information to classify customers into churn and non-churn groups, helping businesses identify customers who may require retention efforts.

Key Points

Binary classification problem.
Predicts customer churn.
Supports business decisions.
Improves customer retention.


2. Telco Customer Churn Dataset

The project uses the Telco Customer Churn dataset, which contains customer demographics, subscription information, billing details, and churn status. This dataset serves as the foundation for training and evaluating the machine learning models.

Key Points

Customer information.
Billing details.
Service usage.
Target variable: Churn.


3. Data Cleaning & Preprocessing

The raw dataset is cleaned before model training by handling missing values, encoding categorical variables, and preparing features suitable for machine learning algorithms.

Key Points

Missing value handling.
Data encoding.
Feature preparation.
Clean dataset generation.


4. Feature Engineering

The project extracts meaningful features such as tenure and service-related information to improve prediction accuracy and help models learn customer behavior effectively.

Key Points

Tenure features.
Usage patterns.
Billing features.
Customer service information.


5. Model Training

Multiple machine learning algorithms are trained using the processed dataset to identify the best-performing classification model.

Key Points

Logistic Regression.
Random Forest.
XGBoost.
Best model selection.


6. Model Evaluation

The trained models are evaluated using standard classification metrics to measure predictive performance.

Key Points

ROC-AUC.
Precision.
Recall.
Confusion Matrix.


7. Explainable AI (SHAP)

SHAP is used to explain why the model predicts that a customer will churn. It highlights the importance of individual features for each prediction.

Key Points

Feature importance.
Explain predictions.
Transparent AI.
Business interpretation.


8. Exploratory Data Analysis (EDA)

Before training the model, exploratory data analysis is performed to understand customer behavior, identify trends, and visualize important patterns in the dataset.

Key Points

Dataset exploration.
Feature analysis.
Data visualization.
Pattern identification.


9. Streamlit Web Application

A web interface allows users to enter customer information and obtain churn predictions without running Python scripts manually.

Key Points

User-friendly interface.
Interactive prediction.
Easy deployment.
Business usability.


10. FastAPI Backend

The project includes a REST API that exposes the prediction model so external applications can request predictions programmatically.

Key Points

REST API.
Prediction endpoint.
Backend integration.
Scalable service.


11. Batch Prediction

Instead of predicting one customer at a time, the project supports predicting churn for multiple customers from a file.

Key Points

Bulk predictions.
CSV input.
Faster processing.
Enterprise use.


12. Docker Deployment

Docker support enables the project to run consistently across different operating systems without dependency issues.

Key Points

Portable environment.
Easy deployment.
Dependency isolation.
Containerization.


13. MLflow Integration

MLflow is included for tracking experiments, model versions, and training information, making the machine learning workflow easier to manage.

Key Points

Experiment tracking.
Model versioning.
Performance logging.
Reproducibility.


14. Monitoring

Monitoring utilities help track the health and performance of the deployed model over time.

Key Points

Performance monitoring.
Health tracking.
Reliability.
Maintenance support.


15. Business Insights

The project generates business insights based on churn predictions, helping organizations understand important factors affecting customer retention.

Key Points

Customer behavior.
Business recommendations.
Decision support.
Churn analysis.


16. Business Impact Analysis

Beyond predicting churn, the project estimates the business impact of customer attrition and highlights why retention strategies matter.

Key Points

Revenue impact.
Customer retention.
Business value.
Strategic planning.


17. Feature Importance Visualization

The project visualizes which features contribute the most to churn prediction, making model behavior easier to understand.

Key Points

Important features.
Model interpretation.
Visual analysis.
Decision support.


18. ROC Curve

The Receiver Operating Characteristic (ROC) Curve is generated to evaluate the model's ability to distinguish between churn and non-churn customers.

Key Points

Classification quality.
Threshold analysis.
Model comparison.
Performance evaluation.


19. Confusion Matrix

A confusion matrix is generated to summarize correct and incorrect predictions and analyze classification errors.

Key Points

True positives.
False positives.
False negatives.
True negatives.


20. Saved Machine Learning Model

After training, the best-performing model is stored for future predictions without retraining.

Key Points

Reusable model.
Faster predictions.
Deployment ready.
Model persistence.


21. Retraining Pipeline

The project supports retraining the model when new customer data becomes available, allowing continuous improvement.

Key Points

Model updates.
Continuous learning.
New data integration.
Performance improvement.


22. Sample Data Generation

Utilities are provided to generate sample customer datasets for testing and demonstration purposes.

Key Points

Test datasets.
Demo data.
Faster testing.
Development support.


23. Automated Testing

Unit tests verify that preprocessing, prediction, and other project components work correctly, improving reliability.

Key Points

Unit testing.
Pipeline validation.
Prediction testing.
Error detection.


24. GitHub Actions (CI)

Continuous Integration automatically runs tests whenever changes are made to the project, helping maintain code quality.

Key Points

Automated testing.
Continuous integration.
Code validation.
Workflow automation.


25. Documentation

The project contains multiple documentation files describing fixes, optimization strategies, scalability considerations, and overall project usage.

Key Points

Project guide.
Optimization notes.
Scalability guide.
Technical documentation.
