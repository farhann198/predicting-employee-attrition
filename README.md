# Employee Attrition Prediction using Machine Learning

## Overview

Employee attrition is a major challenge for organizations, leading to increased recruitment costs and reduced productivity. This project develops a machine learning pipeline to predict employee attrition using the IBM HR Analytics dataset. The workflow includes data preprocessing, feature engineering, model training, hyperparameter optimization, survival analysis, and explainable AI techniques to identify employees at risk of leaving.

This work is based on our IEEE-published research on employee attrition prediction.

---

## Dataset

- **Dataset:** IBM HR Analytics Employee Attrition Dataset
- **Records:** 1,470 employees
- **Features:** 35 HR-related attributes
- **Target Variable:** Attrition (Yes/No)

The dataset contains employee demographic information, compensation details, job satisfaction metrics, work environment factors, and career progression indicators.

---

## Objectives

- Predict employee attrition using machine learning.
- Compare multiple classification algorithms.
- Handle class imbalance effectively.
- Explain model predictions using SHAP.
- Estimate employee time-to-attrition using survival analysis.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Optuna
- SHAP
- Imbalanced-learn (SMOTE-Tomek)
- Lifelines (Cox Proportional Hazards)

---

## Workflow

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Label Encoding & One-Hot Encoding
5. Train-Test Split
6. SMOTE-Tomek Class Balancing
7. Hyperparameter Optimization with Optuna
8. Model Training
9. Model Evaluation
10. Survival Analysis
11. SHAP Explainability

---

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- XGBoost (Best Model)

---

## Results

| Metric | Value |
|---------|------:|
| Accuracy | **92.4%** |
| ROC-AUC | **97.6%** |
| Concordance Index (C-index) | **0.95** |
| Hyperparameter Optimization | **Optuna (100 Trials)** |

The XGBoost model achieved the best predictive performance after Bayesian hyperparameter optimization.

---

## Explainable AI

SHAP (SHapley Additive exPlanations) was used to interpret model predictions and identify the most influential features.

Top contributors included:

- Job Satisfaction
- Monthly Income
- Years at Company
- OverTime
- Age

---

## Project Structure

```
├── attrition-prediction.ipynb
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/farhann198/predicting-employee-attrition.git

cd predicting-employee-attrition

pip install -r requirements.txt
```

---

## Run the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
attrition-prediction.ipynb
```

---

## Future Improvements

- Deploy the model using FastAPI or Streamlit
- Add MLflow experiment tracking
- Dockerize the application
- Build an interactive HR dashboard
- Integrate real-time employee data

---

## Research Publication

**Predicting Employee Attrition by Using Machine Learning Techniques**

Published in **IEEE Xplore (2026)**.

DOI:
https://doi.org/10.1109/ICCSC67078.2026.11468643

---

## Author

**Farhan Ahmad Ansari**

- LinkedIn: https://www.linkedin.com/in/farhan0105g
