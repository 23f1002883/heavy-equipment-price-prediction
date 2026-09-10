# Heavy Equipment Lifecycle & Price Prediction

## 📌 Project Overview
This repository contains an end-to-end predictive machine learning model developed to forecast the financial valuation and operational lifecycles of heavy industrial machinery. The project explores a multi-modal dataset consisting of high-cardinality technical configurations, transactional records, and system metadata.

## 🎯 Model Objective & Evaluation
The primary objective of the model is to minimize the **Root Mean Squared Logarithmic Error (RMSLE)** between the predicted machinery valuation and the observed historical target variables. 

## 🛠️ Data Engineering & Modeling Workflow

* **Data Engineering:** Developed fully insulated preprocessing pipelines to clean high-cardinality geographic features and handle systemic missing operational metadata without introducing data leakage.
* **Mathematical Transformations:** Utilized exploratory data analysis to isolate heavily skewed target price distributions, applying log-scale transformations to stabilize variance and normalize data distributions.
* **Ensemble Modeling:** Built and evaluated a mathematically rigorous Ridge Regression baseline before deploying advanced gradient-boosted trees via **LightGBM** and **CatBoost** to accurately model non-linear feature interactions.
* **Hyperparameter Optimization:** Executed targeted cross-validation routines to fine-tune tree depth and regularization parameters, tracking system variance against optimization metrics.

## 🧮 Core Tech Stack
* **Languages:** Python
* **Libraries:** Scikit-Learn, NumPy, Pandas, LightGBM, CatBoost, RidgeRegression
* **Tools:** Jupyter Notebook, Git
