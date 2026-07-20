# 🏦 Credit Intelligence Engine — Automated Risk Assessment System

An end-to-end machine learning pipeline and interactive web dashboard built to evaluate loan applicant creditworthiness using financial and demographic data.

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Machine Learning Architecture](#machine-learning-architecture)
- [Installation & Setup](#installation--setup)
- [System Flow](#system-flow)
- [Model Robustness & Validation](#model-robustness--validation)
- [Project Information](#project-information)

## Project Overview

The **Credit Intelligence Engine** is a production-ready risk profiling system that allows financial operators to input applicant parameters and instantly receive a mathematically backed approval or denial decision.

The system is divided into two major sections:

| Section | Description |
|---|---|
| **Pipeline (Backend)** | Jupyter notebooks handling data cleaning, feature scaling, model training, evaluation metrics (ROC-AUC), and artifact serialization. |
| **Dashboard (Frontend)** | A sleek, responsive Streamlit web application for real-time inference, featuring dynamic probability charts and adjustable risk thresholds. |

## Features

### Machine Learning Pipeline (Notebooks)

- ✅ Complete Exploratory Data Analysis (EDA) with distribution visualizations
- ✅ Strict train/test splitting with class stratification
- ✅ Feature scaling using `StandardScaler` (fitted only on training data to prevent leakage)
- ✅ Evaluation of multiple algorithms (Logistic Regression, Decision Tree, Random Forest)
- ✅ Automated extraction of Feature Importances and Confusion Matrices
- ✅ Serialization of champion models using `joblib`

### Inference Dashboard (User Interface)

- ✅ Clean, modern UI with bounded numeric inputs and dropdowns
- ✅ Expandable calibration panel to adjust the decision threshold in real-time
- ✅ Dynamic metric cards displaying model confidence percentages
- ✅ Automated horizontal bar charts plotting probability distributions
- ✅ High-performance `@st.cache_resource` loading for instant predictions

## Tech Stack

| Technology | Usage |
|---|---|
| **Python 3.10+** | Core programming language |
| **Scikit-Learn** | Machine learning algorithms and preprocessing |
| **Pandas & NumPy** | Data manipulation and matrix operations |
| **Streamlit** | Frontend web dashboard and interactive UI |
| **Matplotlib & Seaborn** | Data visualization and probability plotting |
| **Joblib** | Serialization and deserialization of model artifacts |

## Folder Structure

```
credit-intelligence-engine/
│
├── data/
│   ├── raw/                      # Original credit.csv dataset
│   └── processed/                # Scaled and split matrices ready for training
│
├── models/
│   ├── credit_model.pkl          # Serialized Random Forest champion model
│   └── scaler.pkl                # Serialized StandardScaler artifact
│
├── notebooks/
│   ├── 01_Preprocessing_and_EDA.ipynb  # Data ingestion, cleaning, and scaling
│   ├── 02_Model_Training.ipynb         # Algorithm training, evaluation, and export
│   └── 03_Inference_Pipeline.ipynb     # Testing ground for the deployed model
│
├── frontend.py                   # Main Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Machine Learning Architecture

The system utilizes a serialized artifact architecture to ensure production data is treated exactly like training data:

```
Raw Data (CSV)
      ↓
Preprocessing (Imputation & Scaling) ──→ scaler.pkl (Saved)
      ↓
Model Training (Random Forest)
      ↓
Evaluation (ROC-AUC, F1-Score) ────────→ credit_model.pkl (Saved)
      ↓
Streamlit Frontend loads .pkl files
      ↓
Live Applicant Data ──→ Scaled ──→ Predicted ──→ Dashboard Visuals
```

## Installation & Setup

### Requirements

- Python 3.8 or higher installed on your system
- Git (optional, for cloning)

### Steps

**1. Clone or download the project**

```bash
git clone https://github.com/yourusername/credit-intelligence-engine.git
cd credit-intelligence-engine
```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Generate Model Artifacts**

Before launching the frontend, you must generate the `.pkl` files.

- Open Jupyter Notebook or VS Code.
- Run all cells in `notebooks/01_Preprocessing_and_EDA.ipynb`.
- Run all cells in `notebooks/02_Model_Training.ipynb`.
- Verify that `credit_model.pkl` and `scaler.pkl` now exist in the `/models` folder.

**5. Run the Dashboard**

```bash
streamlit run frontend.py
```

Open your browser and visit: `http://localhost:8501`

## System Flow

```
Operator Visits Dashboard
        ↓
Adjusts Risk Tolerance Threshold (Optional)
        ↓
Enters Applicant Financial & Demographic Data
        ↓
Clicks "Execute Inference Protocol"
        ↓
Data is formatted and scaled via scaler.pkl
        ↓
credit_model.pkl calculates Default vs. Approval Probability
        ↓
Dashboard instantly renders Decision, Confidence Metric, and Chart
```

## Model Robustness & Validation

| Challenge | Solution Implemented |
|---|---|
| **Data Leakage** | Scaler is strictly fit on `X_train` and only applied to `X_test` and live inputs. |
| **Class Imbalance** | Random Forest utilizes `class_weight='balanced'` to heavily penalize missing high-risk defaults. |
| **Input Errors** | Streamlit UI enforces min/max boundaries on numeric inputs to prevent impossible applicant data (e.g., negative age). |
| **Model Bloat** | Streamlit uses `@st.cache_resource` to load the `.pkl` files only once per session, ensuring rapid latency. |

## Project Information

| Field | Details |
|---|---|
| **Domain** | Data Science & Machine Learning Engineering |
| **Project Title** | Credit Intelligence Engine |
| **Technologies** | Python, Scikit-Learn, Streamlit, Pandas |
| **Project Type** | End-to-End ML Pipeline & Full-Stack Dashboard |
