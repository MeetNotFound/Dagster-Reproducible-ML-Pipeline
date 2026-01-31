# Reproducible Machine Learning Pipeline using Dagster

This repository contains a reproducible machine learning pipeline built using **Dagster**.  
The project demonstrates how asset-based orchestration helps avoid common issues with rerunning Jupyter notebooks by enabling dependency tracking and partial re-execution.

---

## Project Overview

Traditional Jupyter notebook workflows often require rerunning the entire notebook when small changes are made, leading to wasted computation and reproducibility issues.  
This project solves that problem by modeling each step of the ML workflow as a **Dagster asset**.

---

## Dataset

- **Breast Cancer Dataset** from `scikit-learn`
- Contains numerical features derived from breast mass measurements
- Binary target variable (malignant / benign)

---

## Pipeline Structure

The pipeline is implemented using Dagster assets:

- **raw_data**  
  Loads and preprocesses the dataset

- **eda_summary**  
  Generates descriptive statistics for exploratory data analysis

- **train_test**  
  Splits the dataset into training and testing sets

- **decision_tree**  
  Trains a Decision Tree classifier

- **random_forest**  
  Trains a Random Forest classifier

- **logistic_regression**  
  Trains a Logistic Regression model

- **knn**  
  Trains a K-Nearest Neighbors classifier

Dagster automatically builds a **Directed Acyclic Graph (DAG)** based on asset dependencies.

---

## Key Features

- Asset-based pipeline design
- Automatic dependency tracking
- Partial re-execution when data changes
- Multiple machine learning models
- Visual pipeline graph and execution history
- Reproducible and auditable ML workflow

---

## Demonstration of Partial Re-Execution

After the initial pipeline execution, the data asset was modified by sampling a subset of the dataset.  
Upon re-materialization, Dagster re-executed **only the dependent assets**, instead of rerunning the entire pipeline.

This demonstrates Dagster’s ability to save computation time and improve reliability.

---

## Performance Comparison

| Approach | Approximate Time |
|--------|------------------|
| Rerunning full notebook | ~25–30 seconds |
| Dagster partial re-run | ~8–10 seconds |

---

## Project Structure

dagster_ml/
├── init.py
├── repo.py
└── assets/
├── init.py
└── pipeline.py

A033_Meet_Pawar.ipynb
README.md


---

## How to Run

1. Install dependencies:
pip install dagster dagster-webserver scikit-learn pandas


2. Start Dagster:
dagster dev -f dagster_ml/repo.py


3. Open the Dagster UI and materialize assets from the Catalog.

---

## Tools & Technologies

- Python
- Dagster
- Scikit-learn
- Google Colab
- Cloudflare Tunnel (for UI access)

---

## Conclusion

This project demonstrates how Dagster enables reproducible machine learning pipelines by tracking data lineage, execution history, and dependencies.  
It significantly improves reliability and efficiency compared to traditional notebook-based workflows.

---

## Author

**Meet Pawar**
