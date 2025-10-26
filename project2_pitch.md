# Project 2 Pitch — Heart Attack Prediction (India)

## 1. Business Problem Scenario

### Business Problem
Cardiovascular diseases are among the leading causes of death in India.  
Early identification of high-risk individuals can help hospitals and insurance companies reduce emergency hospitalizations and improve preventive care.

This project aims to build a machine learning model that predicts the likelihood of a heart attack based on patient health indicators.

### Stakeholders
- Healthcare providers – to identify high-risk patients early and plan interventions  
- Insurance companies – to assess policy risk and prevent costly claims  
- Patients – to receive personalized health recommendations and early warnings  

### Project Goals
- Develop a predictive model to estimate heart attack risk  
- Enable data-driven decision support for clinicians  
- Improve the efficiency of preventive healthcare initiatives  

### Why Machine Learning?
The relationships between medical risk factors (age, blood pressure, cholesterol, etc.) and cardiovascular outcomes are complex and nonlinear.  
A machine learning approach can capture hidden patterns and interactions that traditional statistical methods may miss.

### Dataset Description
- **Source:** Kaggle – Heart Attack Prediction Dataset (India)  https://www.kaggle.com/datasets/ankushpanday2/heart-attack-risk-and-prediction-dataset-in-india?select=heart_attack_prediction_india.csv
- **Size:** approximately 300–350 patient records  
- **Key Variables:** age, gender, blood pressure, cholesterol, fasting blood sugar, heart rate, physical activity, smoking habits, family history, and target variable (`HeartAttack`: 0 = No, 1 = Yes)  
- **Relevance:** directly aligned with the goal of predicting heart attack risk based on lifestyle and medical data  

### Success Criteria
- **Technical Metrics:** Accuracy, F1-Score, ROC-AUC  
- **Business Metrics:** High recall (≥ 0.8) for identifying at-risk patients, reduction in unplanned hospitalizations  

---

## 2. Problem Solving Process

### Data Acquisition and Understanding
- Load data from CSV file  
- Inspect structure, data types, and missing values  
- Explore variable distributions and correlations  
- Visualize data using histograms, boxplots, and correlation heatmaps  

### Data Preparation and Feature Engineering
- Handle missing values (imputation/removal)  
- Encode categorical features using One-Hot Encoding  
- Scale numerical features using StandardScaler  
- Build a reproducible scikit-learn Pipeline  
- Split data into train/test sets (80/20)  

### Modeling Strategy
Algorithms to evaluate:
1. Logistic Regression (baseline)  
2. Random Forest Classifier  
3. XGBoost or Gradient Boosting  

Additional setup:
- Cross-validation: 5-fold  
- Hyperparameter tuning: GridSearchCV  
- Evaluation metrics: Accuracy, Recall, Precision, ROC-AUC  

### Results Interpretation and Communication
- Compute and visualize feature importance (SHAP or permutation importance)  
- Plot ROC curve, confusion matrix, and metric comparison  
- Translate technical findings into business insights, e.g.  
  "Patients aged 50+ with high blood pressure and cholesterol are 3× more likely to experience a heart attack."


### Conceptual Framework (Flowchart)

```text
 ┌────────────────────────────┐
 │         Raw Data           │
 │ heart_attack_data.csv      │
 └──────────────┬─────────────┘
                │
                ▼
 ┌────────────────────────────┐
 │       Data Cleaning        │
 │ Handle NaN, outliers,      │
 │ encode categories          │
 └──────────────┬─────────────┘
                │
                ▼
 ┌────────────────────────────┐
 │   Feature Engineering      │
 │ Scaling, transformations,  │
 │ new derived features       │
 └──────────────┬─────────────┘
                │
                ▼
 ┌────────────────────────────┐
 │     ML Pipeline Setup      │
 │ scikit-learn pipeline for  │
 │ reproducibility            │
 └──────────────┬─────────────┘
                │
                ▼
 ┌────────────────────────────┐
 │       Model Training       │
 │ LogisticReg, RF, XGBoost   │
 └──────────────┬─────────────┘
                │
                ▼
 ┌────────────────────────────┐
 │ Cross-Val & Tuning         │
 │ GridSearchCV optimization  │
 └──────────────┬─────────────┘
                │
                ▼
 ┌────────────────────────────┐
 │     Model Evaluation       │
 │ Accuracy, Recall, ROC-AUC  │
 └──────────────┬─────────────┘
                │
                ▼
 ┌────────────────────────────┐
 │  Interpretation & Insights │
 │ SHAP values, feature ranks │
 └──────────────┬─────────────┘
                │
                ▼
 ┌────────────────────────────┐
 │ Visualization & Reporting  │
 │ ROC curves, key metrics    │
 └──────────────┬─────────────┘
                │
                ▼
 ┌────────────────────────────┐
 │ Business Recommendations   │
 │ Preventive actions, risk   │
 │ stratification             │
 └────────────────────────────┘

---


## 3. Timeline and Scope

| Phase                              | Key Tasks                                                | Duration  |
|------------------------------------|----------------------------------------------------------|-----------|
| Dataset Finalization & Probl. Def. | Validate dataset, define business objectives             | 1–2 days  |
| Exploratory Data Analysis (EDA)    | Data profiling, correlations, visualization              | 2 days    |
| Data Preprocessing                 | Cleaning, feature engineering, pipeline setup            | 2 days    |
| Model Development                  | Train three models, perform GridSearchCV                 | 3 days    |
| Model Evaluation & Refinement      | Final model selection, interpretability, test evaluation | 2 days    |
| Documentation & Reporting          | Technical report, executive presentation                 | 2 days    |
| Final Review & Submission          | QA check, video recording, submission                    | 1 day     |

---

## 4. Anticipated Challenges and Learning Goals
- Class imbalance (fewer positive heart attack cases)  
- Managing feature correlation and multicollinearity  
- Learning model interpretability tools (SHAP, LIME)  
- Testing generalizability across different populations  

---

## Outcome
A complete end-to-end machine learning pipeline that predicts the likelihood of heart attacks in Indian patients and translates model results into actionable medical and business insights.

---

## 5. Deliverables
- Jupyter Notebook with full workflow (data preprocessing, modeling, evaluation)
- Markdown or PDF project report
- Executive summary presentation slides
- 5–10 minute video presentation explaining the model and insights

---

## 6. Ethical Considerations

This project uses publicly available, anonymized health data.
No personally identifiable information (PII) is present.
All analyses will follow ethical standards for data handling and model fairness.

All code and results will be stored in a version-controlled GitHub repository to ensure full reproducibility.

