# Customer-Churn-Prediction-
Bank customer churn prediction using ML (Logistic Regression, Random Forest, XGBoost). Includes EDA, feature engineering, model comparison (XGBoost AUC 0.87), and risk-tiered retention strategy mapping to turn predictions into actionable business recommendations.


Project Overview

Customer churn is one of the most expensive problems in retail banking — acquiring a new customer typically costs far more than retaining an existing one, yet most banks only discover a customer has left after the account is already closed. This project builds an end-to-end machine learning pipeline that predicts which customers are likely to churn before it happens, giving retention teams a window to intervene proactively instead of reactively.

The dataset consists of ~10,000 retail bank customers, with an approximate 20% churn rate (Exited = 1). The project follows a complete data science workflow:

Exploratory Data Analysis
Investigated churn distribution, missing values, and duplicates, then analyzed churn patterns across geography, age groups, and account activity. Key findings: Germany shows a disproportionately high churn rate (~32%) despite having the smallest customer base, and churn rises sharply for customers holding either very few or too many products.

Feature Engineering
Cleaned and transformed raw data — encoded categorical variables, binned Age into groups to capture non-linear effects, and prepared features for modeling via one-hot encoding.

Model Development
Trained and compared three classifiers — Logistic Regression, Random Forest, and XGBoost — using a stratified train-test split to preserve class balance. Evaluated each model on Accuracy, Precision, Recall, F1-score, and ROC-AUC, since accuracy alone is misleading on imbalanced data.

Results
XGBoost was the strongest performer, achieving a ROC-AUC of 0.87, outperforming Random Forest (0.85) and Logistic Regression (0.77). The top churn drivers identified were NumOfProducts, IsActiveMember, Age, and Geography_Germany — closely aligned with patterns surfaced during EDA.

From Prediction to Action
Rather than stopping at a model score, each customer was assigned a churn probability and segmented into Low / Medium / High risk tiers. Each tier was mapped to a concrete retention strategy — e.g., dedicated relationship managers and 7-day outreach for high-risk customers, region-specific trust-building campaigns for Germany, and engagement nudges for inactive members — turning the model's output into a decision-ready tool for a retention team.

Tech Stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · Seaborn · Matplotlib
