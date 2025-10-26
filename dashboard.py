
########################################
# Heart Attack Risk Prediction Dashboard
# Streamlit version
########################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import os

# -----------------
# 0. Page config
# -----------------
st.set_page_config(
    page_title="Heart Attack Risk Model Dashboard",
    layout="wide"
)

st.title("🩺 Heart Attack Risk Prediction Dashboard")
st.write("Interactive summary of model performance, fairness, and clinical/business insights.")

st.sidebar.header("About this dashboard")
st.sidebar.markdown(
    """
    This dashboard summarizes:
    - Model quality on held-out test data
    - Error balance (FN vs FP)
    - Fairness across subgroups
    - Clinical/business interpretation

    This model is **not** a diagnostic tool.  
    It is intended for **early risk flagging / triage support**.

    Key metric priority: **Recall on the positive (high-risk) class** to reduce False Negatives.
    """
)

# -----------------
# 1. Load exported artifacts
# -----------------
# We assume you ran the "export block" in the notebook which created a folder `dashboard_data/`
DATA_DIR = "dashboard_data"

metrics_df = None
conf_matrix = None
fairness_df = None
coef_df = None
roc_df = None

try:
    metrics_df = pd.read_csv(os.path.join(DATA_DIR, "metrics.csv"))
except Exception as e:
    metrics_df = None

try:
    conf_matrix = pd.read_csv(os.path.join(DATA_DIR, "confusion_matrix.csv"), index_col=0).values
except Exception as e:
    conf_matrix = None

try:
    fairness_df = pd.read_csv(os.path.join(DATA_DIR, "fairness.csv"))
except Exception as e:
    fairness_df = None

try:
    coef_df = pd.read_csv(os.path.join(DATA_DIR, "coefficients.csv"))
except Exception as e:
    coef_df = None

try:
    roc_df = pd.read_csv(os.path.join(DATA_DIR, "roc_curve.csv"))
except Exception as e:
    roc_df = None


tab_perf, tab_fair, tab_interp = st.tabs(
    ["📈 Model Performance", "⚖️ Fairness & Ethics", "🩺 Clinical Interpretation"]
)

# -----------------
# TAB 1: PERFORMANCE
# -----------------
with tab_perf:
    st.subheader("Overall Model Metrics")

    if metrics_df is not None:
        st.dataframe(
            metrics_df.style.format(
                {c: "{:.2f}" for c in metrics_df.columns if c not in ["model"]}
            )
        )
        st.markdown(
            """
            **Why Recall matters:**  
            Recall on the positive class tells us how many truly high-risk patients we correctly flag.  
            Missing a high-risk patient (False Negative) can mean no early intervention.
            """
        )
    else:
        st.error("No metrics available. Did you export metrics.csv from the notebook?")

    st.markdown("#### Confusion Matrix")
    if conf_matrix is not None:
        fig_cm, ax_cm = plt.subplots(figsize=(4,4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            display_labels=['Low Risk (0)', 'High Risk (1)']
        )
        disp.plot(values_format='d', cmap='Blues', ax=ax_cm, colorbar=False)
        ax_cm.set_title("Confusion Matrix (Test Set)")
        st.pyplot(fig_cm)

        try:
            TN, FP, FN, TP = conf_matrix.ravel()
            st.markdown(
                f"""
                **Interpretation:**  
                - True Negatives (TN): {TN} patients correctly identified as low risk  
                - False Positives (FP): {FP} patients unnecessarily flagged as high risk  
                - False Negatives (FN): {FN} high-risk patients were *missed* (most clinically concerning)  
                - True Positives (TP): {TP} high-risk patients correctly flagged  
                """
            )
        except Exception:
            pass
    else:
        st.error("No confusion_matrix.csv found. Please export confusion matrix.")

    st.markdown("#### ROC Curve (Test Set)")
    if roc_df is not None and {"fpr","tpr","auc"}.issubset(roc_df.columns):
        fig_roc, ax_roc = plt.subplots(figsize=(4,4))
        ax_roc.plot(roc_df["fpr"], roc_df["tpr"], label=f"AUC = {roc_df['auc'].iloc[0]:.2f}")
        ax_roc.plot([0,1],[0,1],'--',color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate (Recall)")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

        st.info(
            "AUC describes class separability. AUC close to 1 is ideal. "
            "Lower AUC means risk classes overlap and the model is best used as a triage aid, "
            "not a final diagnostic system."
        )
    else:
        st.warning("roc_curve.csv not found or missing columns fpr,tpr,auc.")

# -----------------
# TAB 2: FAIRNESS
# -----------------
with tab_fair:
    st.subheader("Subgroup Error Analysis")

    st.write(
        """
        We break down model performance by subgroup to check for potential bias.

        Ethical interpretation:
        - High **False Negative** rate in a subgroup → that group is being under-protected.
        - High **False Positive** rate in a subgroup → that group may face unnecessary stress/testing.
        """
    )

    if fairness_df is not None:
        if "Recall_in_group" not in fairness_df.columns and \
           {"TP","FN"}.issubset(fairness_df.columns):
            fairness_df["Recall_in_group"] = fairness_df["TP"] / (fairness_df["TP"] + fairness_df["FN"])

        if "Precision_in_group" not in fairness_df.columns and \
           {"TP","FP"}.issubset(fairness_df.columns):
            fairness_df["Precision_in_group"] = fairness_df["TP"] / (fairness_df["TP"] + fairness_df["FP"])

        st.dataframe(
            fairness_df[["Group","TN","FP","FN","TP","Recall_in_group","Precision_in_group"]]
            .style.format({"Recall_in_group":"{:.2f}", "Precision_in_group":"{:.2f}"})
        )

        st.markdown(
            """
            **Observations:**
            - Gender groups show broadly similar recall, suggesting no extreme gender-specific failure mode.
            - Recall tends to improve in older age groups (60+), which is clinically plausible
              because classic cardiometabolic risk factors are stronger and easier to detect.
            - No single subgroup appears catastrophically underserved, which is encouraging.

            ⚠ Location-based features (state, emergency response time) are influential. This may reflect
            structural healthcare inequality, not just biology. That must be reviewed before deployment.
            """
        )

        st.warning(
            "Use case policy: this model should support preventive outreach and triage, "
            "NOT deny services or insurance coverage to specific groups."
        )
    else:
        st.error("No fairness.csv found. Export subgroup metrics from the notebook.")

# -----------------
# TAB 3: INTERPRETATION
# -----------------
with tab_interp:
    st.subheader("Which Features Drive Predicted Risk?")

    st.markdown(
        """
        We use a regularized Logistic Regression model, which is inherently interpretable.
        - Positive coefficient → pushes prediction toward **high risk**
        - Negative coefficient → pushes prediction toward **low risk / protective**
        """
    )

    if coef_df is not None and {"feature","coefficient"}.issubset(coef_df.columns):
        coef_sorted = coef_df.reindex(
            coef_df["coefficient"].abs().sort_values(ascending=False).index
        ).head(15)

        fig_coef, ax_coef = plt.subplots(figsize=(6,4))
        ax_coef.barh(coef_sorted["feature"][::-1], coef_sorted["coefficient"][::-1])
        ax_coef.set_title("Top Logistic Regression Coefficients")
        ax_coef.set_xlabel("Coefficient (positive = higher predicted risk)")
        st.pyplot(fig_coef)

        st.markdown(
            """
            **Clinical / policy interpretation:**
            - Higher predicted risk is associated with certain regions, delayed emergency response,
              older age, and elevated stress.
            - Protective signals include access to insurance and healthcare, and non-smoking behavior.

            The model is capturing both biology (age, cardiometabolic stress) and
            social determinants of health (access, environment). This is valuable for resource planning.
            """
        )
    else:
        st.error("No coefficients.csv found with columns ['feature','coefficient'].")

    st.info(
        "This dashboard is intended for care teams / public health planners to prioritize outreach.\n"
        "It is NOT a standalone diagnostic model."
    )
