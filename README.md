# TransactSense: Hybrid Financial Fraud Detection Engine

## Overview
TransactSense is an end-to-end, containerized Machine Learning pipeline engineered to detect multi-dimensional financial fraud across both credit card transactions (Sparkov dataset) and mobile money transfers (PaySim dataset). 

This system moves beyond traditional rule-based logic by implementing a **Hybrid Machine Learning Architecture**. It stacks Unsupervised Learning (Behavioral Clustering) and Semi-Supervised Learning (Anomaly Detection) as contextual feature inputs into a highly tuned Supervised ensemble classifier (XGBoost). The inference engine is exposed via a FastAPI microservice and containerized with Docker for seamless cloud deployment.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Directory Structure](#directory-structure)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Model Performance & Interpretability](#model-performance--interpretability)
6. [Local Deployment & Containerization](#local-deployment--containerization)
7. [API Reference](#api-reference)

---

## System Architecture

The pipeline processes high-variance spatio-temporal data and sequential ledger data through a five-stage architecture:

1. **Feature Engineering:** Translates raw transaction data into mathematical signals (e.g., Haversine geographical distances, 24-hour transaction velocity, and ledger depletion ratios).
2. **Contextual Baselining (K-Means):** Aggregates user history to dynamically assign a Behavioral Cluster, providing relative baseline context to the classifier.
3. **Deviation Detection (Isolation Forest):** Trained strictly on legitimate (`is_fraud == 0`) data to map the geometric shape of nominal behavior, outputting a continuous anomaly score for every live transaction.
4. **Supervised Classification (XGBoost):** An Extreme Gradient Boosting ensemble trained on the enriched feature set (Raw Data + Clusters + Anomaly Scores). Balanced using SMOTE to handle extreme minority class ratios.
5. **Real-Time API (FastAPI):** Exposes the serialized model state via a RESTful endpoint with strict Pydantic V2 data validation and dynamic risk-tier routing.

---

## Technology Stack

* **Data Engineering & Analysis:** Python, Pandas, NumPy
* **Machine Learning & Modeling:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
* **Model Interpretability:** SHAP (SHapley Additive exPlanations)
* **API Framework:** FastAPI, Pydantic, Uvicorn
* **DevOps & Deployment:** Docker, joblib

---

## Directory Structure

```text
FraudDetection_project/
├── api/
│   ├── Dockerfile                 # Container blueprint for the microservice
│   ├── main.py                    # FastAPI server & business logic routing
│   ├── requirements.txt           # Dependency lockfile for production
│   └── xgb_paysim_model.joblib    # Serialized XGBoost model payload
├── notebooks/                     
│   ├── Day1_EDA.ipynb             # Exploratory Data Analysis & Feature Identification
│   ├── Day3_Feature_Eng.ipynb     # Spatial, Temporal, and Ledger Logic Engineering
│   ├── Day5_KMeans.ipynb          # Unsupervised Profiling & Clustering
│   ├── Day6_IsolationForest.ipynb # Semi-supervised Anomaly Detection
│   ├── Day7_Classifiers.ipynb     # XGBoost Training, SMOTE integration
│   ├── Day8_Evaluation.ipynb      # Precision/Recall tuning & SHAP interpretability
│   └── Day10_Docker.ipynb         # Deployment staging and API formulation
├── outputs/                       # Evaluation charts (Confusion Matrices, SHAP plots)
├── datasets/                      # Raw Sparkov and PaySim datasets (Ignored in Git)
└── saved_models/                  # Serialized state objects (Scalers, Encoders) (Ignored in Git)

Machine Learning Pipeline
1. Data Preprocessing
Temporal Splitting: Data is split strictly chronologically (or 80/20 train-test) to prevent data leakage and simulate real-world unseen data.

Encoding & Scaling: Handled via Scikit-Learn’s OrdinalEncoder (handling unknown variables dynamically) and StandardScaler for distance-based calculations.

Resampling: SMOTE (Synthetic Minority Over-sampling Technique) is applied exclusively to the training set to achieve a 1:10 minority class ratio, stabilizing gradient descent without corrupting test validity.

2. Context Generation
K-Means: Identifies optimal customer segments (K=5) to prevent global thresholds from misclassifying localized behavior (e.g., wealthy accounts vs. standard accounts).

Isolation Forest: Calculated with contamination=0.01 against purely legitimate data to mathematically define the boundary of nominal transactions.

3. Classification
XGBoost Classifier: Tree-based ensemble capable of parsing non-linear interactions between raw financial inputs and the newly generated contextual features (customer_cluster, anomaly_score).

Model Performance & Interpretability
The system handles the Precision/Recall trade-off dictated by financial sector requirements: prioritizing the capture of zero-day threats while managing false-positive operational costs.

Sparkov Classifier: Achieved an 87% Recall on unseen 2020 testing data, utilizing a heavily weighted loss function (scale_pos_weight) to prioritize identifying fraudulent vectors.

PaySim Classifier: Tuned to achieve near-perfect Recall. Post-evaluation tuning removed artificial algorithmic hypersensitivity, restoring optimal Precision and eliminating false-positive bloat.

Explainability: Integrated SHAP (SHapley Additive exPlanations) to analyze model outputs. This allows compliance teams to mathematically quantify the exact contribution of specific features (such as distance_km and anomaly_score) to any given block/approve decision.

Local Deployment & Containerization
This project is fully containerized. To initialize the inference server on a local environment:

1. Clone the repository:
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME/api

2. Build the Docker Image:
docker build -t fraud-engine-api .

3. Run the Container:
docker run -p 8000:8000 fraud-engine-api

4. Access the API:
Health Check: Navigate to http://localhost:8000/

Interactive Swagger UI: Navigate to http://localhost:8000/docs to test live JSON payloads.

API Reference
Endpoint: POST /predict/paysim

Description: Accepts a JSON payload representing a mobile money transfer, aligns the schema with the serialized model, and returns a probability score alongside a deterministic risk tier.

Input Payload (JSON):

JSON
{
  "amount": 14500.00,
  "balance_drop_ratio": 1.0,
  "txn_velocity": 12,
  "is_transfer_or_cashout": 1,
  "balance_drained": 1,
  "receiver_balance_unchanged": 1
}
Output Response (JSON):

JSON
{
  "transaction_authorized": false,
  "fraud_probability": 0.9999,
  "risk_tier": "CRITICAL - FREEZE ACCOUNT",
  "model_version": "xgb_paysim_tuned_v1"
}