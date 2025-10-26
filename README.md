# ❤️ Heart Attack Risk Prediction (India Dataset)

### 🧠 Overview
This project develops a **machine learning model** to predict **heart attack risk** using demographic, lifestyle, and clinical factors.  
It is built as part of the **Flatiron School Data Science program (Capstone Project 2)**.

The workflow includes:
- Data preprocessing and feature engineering
- Model training with Logistic Regression, Random Forest, and Gradient Boosting
- Fairness and subgroup analysis (Gender, Age Group)
- Clinical interpretation using model coefficients
- Interactive visualization with a **Streamlit dashboard**

---

### 🧩 Project Structure
├── project2_heart_attack.ipynb     # Main Jupyter Notebook
├── dashboard.py                    # Streamlit interactive dashboard
├── dashboard_data/                 # Exported CSV files for dashboard
├── figures/                        # Generated plots
├── heart_attack_prediction_india.csv  # Dataset
└── project2_pitch.md               # Summary and notes

---

### 🚀 How to Run the Dashboard
1. Install dependencies:
   ```bash
   pip install streamlit pandas matplotlib seaborn scikit-learn imbalanced-learn
2.	Run the app:
   streamlit run dashboard.py
3.	Open your browser at: http://localhost:8501

⸻

📊 Key Insights
	•	Model optimized for recall to minimize False Negatives (missed high-risk patients)
	•	Overall balanced model performance (AUC ≈ 0.49)
	•	Age group 60+ and Hypertension are among key risk indicators
	•	Streamlit dashboard allows subgroup-level exploration (Fairness & Ethics)

⸻

🧭 Repository Contents
	•	Notebook – complete analysis pipeline
	•	Dashboard – interactive visual summary
	•	Figures – all major plots (confusion matrix, ROC, feature importance)
	•	Data exports – metrics and coefficients for reproducibility

⸻

👩‍⚕️ Clinical Note

This tool is not a diagnostic engine.
It is designed as an early-warning and triage support system for prioritizing patients at higher risk.

🧠 Author

Olga Chernis
Data Scientist | Flatiron School
📧 grishina.olga.alex@gmail.com

---
