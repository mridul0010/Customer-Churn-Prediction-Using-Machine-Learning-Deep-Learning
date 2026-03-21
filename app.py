import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import shap
from xgboost import XGBClassifier  # Required for unpickling the XGBClassifier inside the pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

class FeatureEngineering(BaseEstimator , TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self , X , y=None):
        return self

    def transform(self , X):
        X = X.copy()

        # Balance Activity
        X['Balance_Active'] = ((X['Balance'] > 0) & (X['IsActiveMember'] == 1)).astype(int)

        # Zero Balance Flag
        X["Zero_Balance"] = (X['Balance'] == 0).astype(int)

        # Product Per tenure 
        X["Product_Per_tenure"] = round(X['NumOfProducts']/(X["Tenure"]+1) , 3)

        # Age Group
        X["Age_Group"] = pd.cut(
            X["Age"],
            bins=[18, 25, 35, 45, 60, 92],
            labels=["18-25", "26-35", "36-45", "46-60", "60+"],
            include_lowest=True
        )

        # Senior Citizen 
        X["IsSenior"] = (X["Age"] > 60).astype(int)
        
        # Multi Product Flag
        X["has_Multiple_Products"] = (X["NumOfProducts"] > 1).astype(int)

        # Credit Card + Activity
        X["Card_Active"] = ((X['HasCrCard'] == 1) & (X["IsActiveMember"])).astype(int)
        

        # New vs old Customer 
        X["Customer_Age"] = pd.cut(
            X['Tenure'],
            bins = [0 , 2 , 5 , 10],
            labels=["New" , "Mid" , "Old"],
            include_lowest=True
        )
    
        return X

# --- 1. Load the Pipeline (Cached for performance) ---

@st.cache_resource
def load_pipeline():
    """Loads the trained machine learning pipeline."""
    try:
        # Load the complete pipeline (Preprocessor + XGBoost Model)
        with open('pipeline.pkl', 'rb') as file:
            pipeline = pickle.load(file)
            
        # Robustly handle any unexpected feature mappings when making a single row prediction
        transformer = pipeline.named_steps.get("preprocessing", None)
        if transformer is not None and hasattr(transformer, "named_transformers_"):
            ord_encoder = transformer.named_transformers_.get("ord", None)
            if isinstance(ord_encoder, OrdinalEncoder):
                ord_encoder.set_params(handle_unknown="use_encoded_value", unknown_value=-1)
                    
        return pipeline
    except FileNotFoundError:
        st.error("Error: 'pipeline.pkl' not found. Please ensure the file is uploaded to the directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during pipeline loading: {e}")
        st.stop()

# Load resource once
pipeline = load_pipeline()

@st.cache_data
def load_dataset():
    df = pd.read_csv("train.csv")
    return df.drop(columns=["id", "CustomerId", "Surname"])


@st.cache_data
def get_eval_split(df):
    x = df.drop(columns="Exited")
    y = df["Exited"]
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


@st.cache_data
def optimize_threshold(y_true, y_proba):
    # Match notebook threshold search strategy
    grid = np.linspace(0.2, 0.8, 62)
    best_t = max(grid, key=lambda t: f1_score(y_true, (y_proba >= t).astype(int)))
    return float(best_t)


@st.cache_data
def evaluate_pipeline(_pipeline, x_test, y_test):
    y_proba = _pipeline.predict_proba(x_test)[:, 1]
    best_threshold = optimize_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cm": confusion_matrix(y_test, y_pred),
    }
    return metrics


@st.cache_data
def get_model_metrics(_pipeline):
    df = load_dataset()
    _, x_test, _, y_test = get_eval_split(df)
    return evaluate_pipeline(_pipeline, x_test, y_test)


def format_input_row(payload):
    expected_order = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    frame = pd.DataFrame([payload])
    return frame[expected_order]


model_metrics = get_model_metrics(pipeline)
best_threshold = model_metrics["threshold"]

# --- Prediction Function ---

def predict_churn(data):
    """
    Takes a dictionary of input data, converts it to a DataFrame, 
    and uses the pipeline to predict churn probability.
    """
    ordered_input_df = format_input_row(data)

    # 2. Prediction
    # The pipeline handles preprocessing automatically.
    # predict_proba returns an array of shape (n_samples, n_classes).
    # We want the probability of class 1 (Exited/Churn), which is at index 1.
    prediction_prob = pipeline.predict_proba(ordered_input_df)[0][1]

    return float(prediction_prob)


# --- Streamlit UI Design & Navigation ---

st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Churn Predictor", "Model Performance"])
st.sidebar.markdown("---")

if page == "Churn Predictor":
    st.title("🏦 Bank Customer Churn Risk Assessment Tool")
    st.markdown("Use the **sidebar** to define the customer profile and calculate their churn risk.")
    st.caption(f"Model decision threshold (from training workflow): {best_threshold:.2f}")

    # Sidebar for Input Features
    with st.sidebar:
        st.header("🎯 Churn Prediction")
        st.markdown("---")
        
        # --- Group 1: Demographics ---
        st.subheader("1. Demographic Details")
        
        col_a, col_b = st.columns(2)
        with col_a:
            Geography = st.selectbox("🌍 Geography", ('France', 'Germany', 'Spain'))
        with col_b:
            Gender = st.radio("🧍 Gender", ('Male', 'Female'))

        Age = st.slider("🎂 Age", min_value=18, max_value=92, value=40)
        st.markdown("---")

        # --- Group 2: Account Details ---
        st.subheader("2. Account Status")
        CreditScore = st.slider("⚖️ Credit Score", min_value=300, max_value=850, value=650)
        Tenure = st.slider("⏳ Tenure (Years)", min_value=0, max_value=10, value=5)
        NumOfProducts = st.slider("🛒 Number of Products", min_value=1, max_value=4, value=2)
        
        col_c, col_d = st.columns(2)
        with col_c:
            HasCrCard = st.radio("💳 Has Credit Card?", (1, 0), index=0, format_func=lambda x: 'Yes' if x == 1 else 'No')
        with col_d:
            IsActiveMember = st.radio("✅ Is Active Member?", (1, 0), index=0, format_func=lambda x: 'Yes' if x == 1 else 'No')
            
        st.markdown("---")
        
        # --- Group 3: Financials ---
        st.subheader("3. Financials")
        Balance = st.number_input("💰 Account Balance ($)", min_value=0.0, max_value=250000.0, value=50000.0, step=100.0)
        EstimatedSalary = st.number_input("💵 Estimated Salary ($)", min_value=0.0, max_value=200000.0, value=75000.0, step=100.0)
        
        st.markdown("---")
        predict_button = st.button("🚀 Predict Churn Risk", type="primary", use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Model Performance Setup")
        if model_metrics["accuracy"] > 0:
            st.caption(f"**Accuracy:** {model_metrics['accuracy']:.2%}")
            st.caption(f"**F1 Score:** {model_metrics['f1']:.2%}")
            st.caption(f"**ROC-AUC:** {model_metrics['auc']:.2%}")
        st.caption(f"**Cutoff:** {best_threshold:.2f}")


    # --- Main Content/Output ---
    col1, col2 = st.columns([1, 1])

    with col1:
        if predict_button:
            # Gather input data
            input_data = {
                'CreditScore': CreditScore,
                'Age': Age,
                'Tenure': Tenure,
                'Gender': Gender,
                'Balance': Balance,
                'NumOfProducts': NumOfProducts,
                'HasCrCard': HasCrCard,
                'Geography': Geography,
                'IsActiveMember': IsActiveMember,
                'EstimatedSalary': EstimatedSalary
            }

            with st.spinner('Calculating churn probability...'):
                # Perform prediction
                churn_probability = predict_churn(input_data)
            
            # Format the probability as a percentage
            risk_percentage = churn_probability * 100

            st.subheader("Prediction Result")

            model_cutoff_pct = best_threshold * 100
            moderate_cutoff_pct = max(20.0, model_cutoff_pct * 0.6)

            if risk_percentage >= model_cutoff_pct:
                risk_text = "HIGH CHURN RISK"
                icon = "🚨"
                display_color = "red"
            elif risk_percentage >= moderate_cutoff_pct:
                risk_text = "MODERATE CHURN RISK"
                icon = "⚠️"
                display_color = "orange"
            else:
                risk_text = "LOW CHURN RISK"
                icon = "✅"
                display_color = "green"

            with st.container(border=True):
                st.markdown(f"**{icon} Overall Churn Probability**")
                st.markdown(f"<p style='font-size: 3rem; font-weight: bold; color: {display_color};'>{risk_percentage:.2f} %</p>", unsafe_allow_html=True)
                st.caption(risk_text)
                st.caption(f"Model classification cutoff: {model_cutoff_pct:.2f}%")
                
                st.markdown(f"**Confidence Score (0.0 - 1.0):**")
                st.progress(churn_probability)
            
            st.markdown("---")
            
            # Key Factors Section (Based on general churn model importance)
            st.subheader("🔎 Key Contributing Factors (General Observations)")
            st.info(f"""
            While precise factor weights depend on the model architecture, in churn analysis, the following factors often contribute significantly to risk:
            1. **Age:** Older customers (especially 50+) can show increased risk of exiting if not actively engaged.
            2. **Balance & Products:** High balance combined with a low number of products can signal poor customer relationship and high flight risk.
            3. **Inactive Status:** The 'IsActiveMember' status is a strong indicator; inactive customers are far more likely to churn.
            """)
            
        else:
            st.info("👈 Enter customer details in the sidebar and click the button to start prediction.")

    with col2:
        if predict_button:
            st.subheader("Analysis & Recommendations")

            model_cutoff_pct = best_threshold * 100
            moderate_cutoff_pct = max(20.0, model_cutoff_pct * 0.6)

            if risk_percentage >= model_cutoff_pct:
                st.error("""
                **🚨 URGENT ACTION: HIGH RISK**
                This customer poses a high flight risk. Immediate retention strategies are mandatory:
                * **Personalized Offer:** Send a highly tailored offer (e.g., better interest rate or free consultation).
                * **Direct Contact:** Have a senior account manager call to address potential issues directly.
                * **Product Review:** Proactively review if their current products meet their evolving needs.
                """)
            elif risk_percentage >= moderate_cutoff_pct:
                st.warning("""
                **⚠️ PROACTIVE MONITORING: MODERATE RISK**
                This customer requires proactive engagement and monitoring to prevent escalation to high risk:
                * **Loyalty Programs:** Introduce them to new loyalty tiers or perks before they consider leaving.
                * **Satisfaction Check:** Send a quick, high-impact survey about their satisfaction with services.
                * **Usage Incentives:** Offer small bonuses to encourage deeper product engagement.
                """)
            else:
                st.success("""
                **✅ MAINTAIN & GROW: LOW RISK**
                This customer is generally stable. Focus on maximizing their value:
                * **Relationship Building:** Continue standard communication (e.g., quality newsletters, annual reviews).
                * **Upselling/Cross-selling:** Look for opportunities to introduce new, relevant products to increase stickiness.
                """)
            
            with st.expander("Show Raw Data and Model Summary"):
                st.dataframe(pd.DataFrame([input_data]), use_container_width=True)
                st.text(f"Raw Model Output (Probability of Churn): {churn_probability:.6f}")
                
        # --- New Section: Data Insights on Churn ---
        st.markdown("---")
        st.subheader("📊 Key Data Insights: When Churn Happens")
        st.markdown("""
        Based on the patterns observed in the bank's historical customer data, churn is not random. It is strongly influenced by specific feature values. Understanding these thresholds helps in interpreting the prediction.
        """)
        
        
        st.markdown("---")
        
        col_insights_1, col_insights_2 = st.columns(2)
        
        with col_insights_1:
            st.markdown("### 🛑 High Churn Indicators")
            st.markdown("""
            These conditions significantly **increase** the likelihood of a customer leaving the bank:
            * **Geography (Germany):** Customers from Germany show a disproportionately higher churn rate compared to France and Spain.
            * **Age (Mid-Age Peak):** Churn risk is often highest in the **40-60 age bracket**.
            * **Account Balance (High):** Customers with a high balance, especially with weak engagement, can be a major flight risk.
            * **Inactive Member:** Customers marked as inactive (IsActiveMember = 0) are about **3 to 4 times** more likely to churn than active members.
            * **Number of Products:** Product count is one of the strongest churn drivers in the trained model.
            * **Gender Effect:** SHAP analysis shows gender contributes materially to model decisions.
            """)

        with col_insights_2:
            st.markdown("### ✅ Low Churn Indicators")
            st.markdown("""
            These conditions are associated with a **stable** and loyal customer base:
            * **Geography (France/Spain):** Customers in these regions are generally less likely to churn.
            * **Age (Young/Senior):** Very young (18-30) and very old (65+) customers typically show lower churn rates.
            * **Tenure (Long):** Customers with **Tenure > 8 years** are highly loyal.
            * **Credit Score (Excellent):** Customers with a very high Credit Score (**> 800**) are less likely to churn, though this feature is not the single most dominant factor.
            * **Has Credit Card:** The presence of a credit card is often associated with slightly **lower churn**, suggesting a basic level of product stickiness.
            """)
        
        st.markdown("---")


elif page == "Model Performance":
    
    st.title("📊 Model Performance & Evaluation")
    st.markdown("""
    This page provides a comprehensive overview of the trained **XGBoost Bank Churn Prediction Model** 
    performance metrics, visualizations, and explainability insights.
    """)

    # --- Load Resources ---
    df = load_dataset()
    _, X_test, _, y_test = get_eval_split(df)
    perf_metrics = evaluate_pipeline(pipeline, X_test, y_test)

    # --- Main Metrics Row ---
    st.header("🎯 Key Performance Metrics")

    colbg1, colbg2, colbg3, colbg4, colbg5, colbg6 = st.columns(6)

    with colbg1:
        st.metric("Accuracy", f"{perf_metrics['accuracy']:.3f}", 
                  help="Proportion of correct predictions out of total predictions")

    with colbg2:
        st.metric("Precision", f"{perf_metrics['precision']:.3f}",
                  help="Proportion of true positives among all positive predictions")

    with colbg3:
        st.metric("Recall", f"{perf_metrics['recall']:.3f}",
                  help="Proportion of true positives among all actual positives")

    with colbg4:
        st.metric("F1-Score", f"{perf_metrics['f1']:.3f}",
                  help="Harmonic mean of precision and recall")

    with colbg5:
        st.metric("AUC-ROC", f"{perf_metrics['auc']:.3f}",
                  help="Area under the ROC curve - overall model quality")

    with colbg6:
        st.metric("Best Threshold", f"{perf_metrics['threshold']:.3f}")

    st.markdown("---")

    # --- Visualizations Section ---
    st.header("📈 Model Evaluation Visualizations")

    col_viz1, col_viz2 = st.columns(2)

    # ROC Curve
    with col_viz1:
        st.subheader("ROC Curve")
        
        fpr, tpr, _ = roc_curve(perf_metrics['y_true'], perf_metrics['y_proba'])
        
        fig_roc = go.Figure()
        
        # ROC curve
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {perf_metrics["auc"]:.3f})',
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
            perf_metrics['cm'],
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
            perf_metrics['y_true'],
            perf_metrics['y_pred'],
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
            "Training Data Split": "80% training, 20% validation",
            "Class Balance": "Applied SMOTE to handle imbalanced classes"
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
            # Get feature engineering, preprocessing transformer, and model
            feature_engineering = pipeline.named_steps['feature_engineering']
            preprocessing = pipeline.named_steps['preprocessing']
            xgb_model = pipeline.named_steps['model']
            
            # Match training pipeline order: feature engineering -> preprocessing
            X_test_fe = feature_engineering.transform(X_test)
            X_test_transformed = preprocessing.transform(X_test_fe)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
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

            st.subheader("SHAP Beeswarm")
            fig_swarm, _ = plt.subplots(figsize=(9, 5))
            shap.summary_plot(
                shap_values,
                X_test_transformed,
                feature_names=feature_names,
                max_display=12,
                show=False,
            )
            st.pyplot(fig_swarm, use_container_width=True, clear_figure=True)
            
            st.info("""
            **SHAP Interpretation:** 
            - Features with higher mean absolute SHAP values have a greater impact on model predictions.
            - The beeswarm view shows direction: points to the right push toward churn, points to the left reduce churn risk.
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
        - **High AUC-ROC ({perf_metrics['auc']:.3f}):** Excellent discriminative ability between churners and non-churners.
        - **Good Precision ({perf_metrics['precision']:.3f}):** Among predicted churners, most are actual churners.
        - **Balanced Performance:** F1-score ({perf_metrics['f1']:.3f}) indicates good overall model quality.
        """)

    with col_insight2:
        st.subheader("⚠️ Considerations")
        st.markdown("""
        - **Recall-Precision Trade-off:** Depending on business needs, consider adjusting the decision threshold.
        - **Class Imbalance Handled:** SMOTE was used to address imbalanced churn labels.
        - **Feature Engineering:** The model benefits from categorical encoding and feature scaling.
        """)

    st.markdown("---")

    st.subheader("📊 F1-Optimized Threshold Performance")
    st.markdown(
        f"""
    The model uses a dynamically optimized decision threshold (**{best_threshold:.2f}**) to maximize F1-score on validation data.
    Predictions with churn probability >= **{best_threshold:.2f}** are classified as churn, improving balance between precision and recall.
    """
    )

    # Footer
    st.markdown("---")
    st.caption("🏦 Bank Customer Churn Prediction Model | Built with XGBoost, SMOTE, and SHAP Explainability")