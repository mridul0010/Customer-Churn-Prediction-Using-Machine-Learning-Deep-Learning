<div align="center">

# 🏦 Customer Churn Prediction

### Machine Learning & Deep Learning Approach

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://customer-churn-prediction-using-machine-learning-deep-learning.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-blue)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ANN-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Predict bank customer churn with an end-to-end ML pipeline — from data preprocessing and model comparison to SHAP-based explainability and interactive deployment.**

[Live Demo](https://customer-churn-prediction-using-machine-learning-deep-learning.streamlit.app/) · [Report Bug](https://github.com/mridul0010/Customer-Churn-Prediction-Using-Machine-Learning-Deep-Learning/issues) · [Request Feature](https://github.com/mridul0010/Customer-Churn-Prediction-Using-Machine-Learning-Deep-Learning/issues)

</div>

---

## 📑 Table of Contents

- [About the Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Project Structure](#-project-structure)
- [Model Development Workflow](#-model-development-workflow)
- [Model Evaluation Metrics](#-model-evaluation-metrics)
- [Model Explainability (SHAP)](#-model-explainability-shap)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Key Takeaways](#-key-takeaways)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 📌 About the Project

This repository implements a **customer churn prediction system** for a banking dataset using a **comparative machine learning approach**. Multiple models were trained and evaluated — **XGBoost was selected as the final production model**, while an **Artificial Neural Network (ANN)** was implemented as a benchmark to demonstrate deep learning understanding.

A **Streamlit web app** is provided for interactive use, allowing users to input customer details and receive a real-time churn risk assessment.

> **Philosophy:** This project emphasizes _model comparison, data-driven model selection, interpretability, and deployment readiness_ — rather than focusing on accuracy alone.

---

## 🧠 Problem Statement

Customer churn significantly impacts business revenue. The objective is to **predict whether a customer is likely to leave the bank**, enabling proactive retention strategies.

| Challenge | Description |
|---|---|
| **Imbalanced Data** | Significant skew between churn vs. non-churn classes |
| **Recall-Focused** | Correctly identifying churned customers is critical |
| **Interpretability** | Business stakeholders require explainable predictions |

---

## 📂 Project Structure

```
Customer-Churn-Prediction/
│
├── Model_Selection.ipynb            # Model comparison & selection notebook
├── Churn_Modelling_Training.ipynb   # Model training notebook (incl. ANN)
├── train.csv                        # Dataset used for training
├── pipeline.pkl                     # Preprocessing + Model pipeline
├── app.py                           # Streamlit web application
├── requirements.txt                 # Project dependencies
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

---

## 🧩 Model Development Workflow

### 1. Data Preprocessing

- Removed non-informative features (Customer ID, Surname)
- Encoded categorical variables (Gender, Geography)
- Scaled numerical features
- Handled class imbalance using **SMOTE**
- Saved preprocessing + model pipeline (`pipeline.pkl`) for consistent inference

### 2. Model Training & Selection

All experiments and comparisons are documented in [`Model_Selection.ipynb`](Model_Selection.ipynb).

| Model | Type | Notes |
|---|---|---|
| **XGBoost** ✅ | Classical ML | Final production model |
| Random Forest | Classical ML | Evaluated as baseline |
| ANN (Keras) | Deep Learning | Benchmark comparison |

All models achieved similar accuracy (~86%), indicating a **performance ceiling driven by feature separability**. Model selection was therefore based on **robustness, interpretability, and suitability for tabular data**.

### 3. Final Model Choice — XGBoost

XGBoost was selected because it:

- ✅ Performs exceptionally well on structured/tabular data
- ✅ Captures non-linear feature interactions efficiently
- ✅ Requires less data and tuning than ANN
- ✅ Provides strong **feature importance and explainability**

> The ANN model is retained as a **benchmark** in [`Churn_Modelling_Training.ipynb`](Churn_Modelling_Training.ipynb), comparing deep learning against classical ML approaches.

---

## 📊 Model Evaluation Metrics

The final **XGBoost model** demonstrates strong performance across multiple evaluation metrics.
The app computes these directly from the saved pipeline using the same evaluation logic in [`app.py`](app.py), including an F1-optimized decision threshold:

### Performance Summary (Test Set)

| Metric | Score | Description |
|---|---|---|
| **Accuracy** | 86.09% | Overall prediction correctness |
| **Precision** | 65.27% | Proportion of correct positive predictions |
| **Recall** | 73.24% | Ability to identify churned customers |
| **F1-Score** | 69.02% | Harmonic mean of precision and recall |
| **AUC-ROC** | 90.53% | Overall model discriminative ability |
| **Best Threshold** | 0.32 | Probability cutoff selected to maximize F1-score |

### F1-Optimized Threshold Strategy

The model uses a **dynamically optimized decision threshold (0.32)** to maximize F1-score on validation data.
Predictions with churn probability **≥ 0.32** are classified as churn, improving the precision-recall balance for this imbalanced problem.

> ⚠️ Given the imbalanced nature of churn prediction, **accuracy was not used as the sole decision metric**. The model prioritizes recall and F1-score to ensure effective identification of at-risk customers.

---

## 🔍 Model Explainability (SHAP)

To improve transparency and business trust, **SHAP (SHapley Additive exPlanations)** was used with the XGBoost model to:

- Identify the most influential features driving churn
- Explain individual predictions
- Support data-driven business decisions

This highlights the importance of **interpretability in real-world ML systems**.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python |
| **Machine Learning** | XGBoost, Random Forest, Scikit-learn |
| **Deep Learning** | TensorFlow, Keras |
| **Explainability** | SHAP |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Deployment** | Streamlit |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/mridul0010/Customer-Churn-Prediction-Using-Machine-Learning-Deep-Learning.git
   cd Customer-Churn-Prediction-Using-Machine-Learning-Deep-Learning
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

4. **Or try the live demo**

   🔗 [https://customer-churn-prediction-using-machine-learning-deep-learning.streamlit.app/](https://customer-churn-prediction-using-machine-learning-deep-learning.streamlit.app/)

---

## 💡 Usage

1. Input customer details in the sidebar (e.g., Age, Geography, Tenure, etc.)
2. Click **"🚀 Predict Churn Risk"**
3. The model returns:
   - **Churn probability** (in %)
   - **Risk level** (Low / Moderate / High)
   - **Data-driven recommendations** for action

---

## 📸 Screenshots

<details>
<summary>Click to expand screenshots</summary>

<br>

<img width="1884" height="1058" alt="image" src="https://github.com/user-attachments/assets/60b0d0fc-f5cc-46e3-9be2-12ab160e224b" />

<img width="1893" height="950" alt="image" src="https://github.com/user-attachments/assets/d1bc87ae-8130-4103-b0c0-bb62b93f91ba" />

<img width="1919" height="978" alt="image" src="https://github.com/user-attachments/assets/5cad3f7f-aabe-4921-8a03-41b13929e2b7" />

<img width="1890" height="983" alt="image" src="https://github.com/user-attachments/assets/a958e238-e73a-44d8-8708-881614073e8c" />

<img width="1848" alt="App Overview" src="https://github.com/user-attachments/assets/add4d33f-75fb-4cef-a4fc-8680218178aa" />

<img width="1908" alt="Prediction Results" src="https://github.com/user-attachments/assets/14ff7043-e82d-4c5d-ae70-cc33f2d7a08d" />

<img width="1919" alt="SHAP Analysis" src="https://github.com/user-attachments/assets/400de439-eec3-4f78-a84f-e50f7ad884fe" />

<img width="1919" alt="Feature Details" src="https://github.com/user-attachments/assets/519a4be4-8101-41f2-b2dd-1124600d9eb1" />

<img width="1866" height="978" alt="image" src="https://github.com/user-attachments/assets/e8989108-3b02-4bb8-9505-f86c7b9f91ff" />

<img width="1827" height="735" alt="image" src="https://github.com/user-attachments/assets/e63b9f5b-e941-4e3d-8917-653c8ce92632" />



</details>

---

## 🎯 Key Takeaways

- Demonstrates **comparative model evaluation** and informed model selection
- Highlights handling of **imbalanced classification problems**
- Balances **ML performance, interpretability, and engineering best practices**
- Shows when deep learning is _not_ the optimal solution for tabular data

---

## 🔮 Future Improvements

- [ ] Implement model explainability dashboard
- [ ] Extend dataset with behavioral transaction data
- [ ] Deploy app on Streamlit Cloud or AWS EC2

---

## 👩‍💻 Author

<table>
<tr>
<td>

**Mridul Lata**

📍 Jaipur, India · 💼 Aspiring Data Scientist / ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mridullata)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/mridul0010)

</td>
</tr>
</table>

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

📌 _This project focuses on real-world ML decision-making, not blind accuracy optimization._

⭐ **If you found this helpful, please give the repository a star and share your feedback!**

</div>
