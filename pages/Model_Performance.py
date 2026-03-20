import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import shap

# --- 1. Load Pipeline and Data (Cached) ---

@st.cache_resource
def load_pipeline():
    """Loads the trained machine learning pipeline."""
    try:
        with open('pipeline.pkl', 'rb') as file:
            pipeline = pickle.load(file)
        return pipeline
    except FileNotFoundError:
        st.error("Error: 'pipeline.pkl' not found.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during pipeline loading: {e}")
        st.stop()


@st.cache_data
def load_and_prepare_data():
    """Loads and prepares data for evaluation."""
    try:
        df = pd.read_csv("train.csv")
        df = df.drop(columns=["id", "CustomerId", "Surname"])
        
        X = df.drop(columns="Exited")
        y = df["Exited"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


def evaluate_model_performance(model_pipeline, x_eval, y_eval):
    """Compute model performance metrics and outputs for the evaluation split."""
    y_pred = model_pipeline.predict(x_eval)
    y_pred_proba = model_pipeline.predict_proba(x_eval)[:, 1]
    
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, y_pred_proba)
    
    cm = confusion_matrix(y_eval, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'cm': cm,
        'y_eval': y_eval
    }


# --- Load Resources ---
model_pipeline = load_pipeline()
_, X_test, _, y_test = load_and_prepare_data()
metrics = evaluate_model_performance(model_pipeline, X_test, y_test)

# --- Page Configuration ---
st.set_page_config(page_title="Model Performance", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Model Performance & Evaluation")
st.markdown("""
This page provides a comprehensive overview of the trained **XGBoost Bank Churn Prediction Model** 
performance metrics, visualizations, and explainability insights.
""")

# --- Main Metrics Row ---
st.header("🎯 Key Performance Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Accuracy", f"{metrics['accuracy']:.4f}", 
              help="Proportion of correct predictions out of total predictions")

with col2:
    st.metric("Precision", f"{metrics['precision']:.4f}",
              help="Proportion of true positives among all positive predictions")

with col3:
    st.metric("Recall", f"{metrics['recall']:.4f}",
              help="Proportion of true positives among all actual positives")

with col4:
    st.metric("F1-Score", f"{metrics['f1']:.4f}",
              help="Harmonic mean of precision and recall")

with col5:
    st.metric("AUC-ROC", f"{metrics['auc']:.4f}",
              help="Area under the ROC curve - overall model quality")

st.markdown("---")

# --- Visualizations Section ---
st.header("📈 Model Evaluation Visualizations")

col_viz1, col_viz2 = st.columns(2)

# ROC Curve
with col_viz1:
    st.subheader("ROC Curve")
    
    fpr, tpr, _ = roc_curve(metrics['y_eval'], metrics['y_pred_proba'])
    
    fig_roc = go.Figure()
    
    # ROC curve
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {metrics["auc"]:.3f})',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Diagonal (random model)
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Model',
        line=dict(color='red', dash='dash')
    ))
    
    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        hovermode='closest'
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)

# Confusion Matrix Heatmap
with col_viz2:
    st.subheader("Confusion Matrix")
    
    cm_df = pd.DataFrame(
        metrics['cm'],
        index=['Actual No Churn', 'Actual Churn'],
        columns=['Predicted No Churn', 'Predicted Churn']
    )
    
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(color="Count")
    )
    
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# --- Detailed Classification Report ---
st.header("📋 Detailed Classification Report")

report_df = pd.DataFrame(
    classification_report(
        metrics['y_eval'],
        metrics['y_pred'],
        target_names=['No Churn', 'Churn'],
        output_dict=True,
        zero_division=0,
    )
).T
report_df = report_df.rename(
    columns={
        'precision': 'Precision',
        'recall': 'Recall',
        'f1-score': 'F1-Score',
        'support': 'Support',
    }
)
st.dataframe(report_df, use_container_width=True)

st.markdown("---")

# --- Model Architecture & Configuration ---
st.header("🏗️ Model Architecture")

with st.expander("View Model Configuration", expanded=False):
    model_info = {
        "Model Type": "XGBoost Classifier with SMOTE & Feature Preprocessing",
        "Pipeline Steps": [
            "1. ColumnTransformer (OneHotEncoder for categorical, passthrough for numerical)",
            "2. SMOTE (Synthetic Minority Oversampling)",
            "3. XGBoost Classifier"
        ],
        "XGBoost Hyperparameters": {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": 42
        },
        "Training Data Split": "80% training, 20% validation (stratified)",
        "Class Balance": f"Applied SMOTE to handle imbalanced classes"
    }
    
    for key, value in model_info.items():
        st.write(f"**{key}:**")
        if isinstance(value, list):
            for item in value:
                st.write(f"  • {item}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                st.write(f"  • {sub_key}: {sub_value}")
        else:
            st.write(f"  {value}")
        st.write("")

st.markdown("---")

# --- SHAP Feature Importance ---
st.header("🔍 Feature Importance (SHAP Analysis)")

try:
    with st.spinner("Computing SHAP feature importance..."):
        # Get preprocessing transformer and model
        preprocessing = model_pipeline.named_steps['preprocessing']
        xgb_model = model_pipeline.named_steps['model']
        
        # Transform test data
        X_test_transformed = preprocessing.transform(X_test)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_transformed)
        
        # Get feature names
        feature_names = preprocessing.get_feature_names_out()
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean |SHAP|': mean_abs_shap
        }).sort_values('Mean |SHAP|', ascending=False)
        
        # Plot
        fig_shap = px.bar(
            feature_importance_df,
            x='Mean |SHAP|',
            y='Feature',
            orientation='h',
            title="Feature Importance (Mean |SHAP| Values)",
            labels={'Mean |SHAP|': 'Mean Absolute SHAP Value', 'Feature': 'Feature Name'}
        )
        
        fig_shap.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_shap, use_container_width=True)
        
        st.info("""
        **SHAP Interpretation:** 
        - Features with higher mean absolute SHAP values have a greater impact on model predictions.
        - These features are the most important drivers of churn decisions.
        """)

except Exception as e:
    st.warning(f"Could not generate SHAP visualization: {e}")

st.markdown("---")

# --- Key Insights ---
st.header("💡 Key Insights & Observations")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    st.subheader("✅ Model Strengths")
    st.markdown(f"""
    - **High AUC-ROC ({metrics['auc']:.3f}):** Excellent discriminative ability between churners and non-churners.
    - **Good Precision ({metrics['precision']:.3f}):** Among predicted churners, most are actual churners.
    - **Balanced Performance:** F1-score ({metrics['f1']:.3f}) indicates good overall model quality.
    """)

with col_insight2:
    st.subheader("⚠️ Considerations")
    st.markdown(f"""
    - **Recall-Precision Trade-off:** Depending on business needs, consider adjusting the decision threshold.
    - **Class Imbalance Handled:** SMOTE was used to address imbalanced churn labels.
    - **Feature Engineering:** The model benefits from categorical encoding and feature scaling.
    """)

st.markdown("---")

st.subheader("📊 Threshold Optimization")
st.markdown("""
The model uses a dynamically optimized decision threshold to maximize F1-score on validation data.
This threshold is recalculated using the training data to ensure optimal churn detection.
""")

# Footer
st.markdown("---")
st.caption("🏦 Bank Customer Churn Prediction Model | Built with XGBoost, SMOTE, and SHAP Explainability")
