# Churn Control

<p align="center">
  <img src="white_logo.png" alt="Churn Control Logo" width="300"/>
</p>

**Churn Control** is a customer churn prediction project that helps businesses identify customers at risk of leaving. The project leverages machine learning models to predict churn probabilities and provide actionable insights.

The service can be accessed it through the website:
https://churn-control.streamlit.app/
---

## Table of Contents
- [Project Overview](#project-overview)
- [Usage](#usage)
- [Metrics and Evaluation](#metrics-and-evaluation)
- [Dataset](#dataset)

---

## Project Overview
Churn Control is designed to give businesses actionable insights into customer retention. The project includes:

- Data preprocessing and feature engineering
- Machine learning model training and evaluation
- An interactive, cloud-hosted Streamlit app that enables users to make predictions and visualize the results.

The goal is to help businesses proactively reduce churn and improve customer satisfaction.

---

## Usage
- Upload customer data through the [website](https://churn-control.streamlit.app/) and receive churn probability predictions
- Visualize feature importance using SHAP plots
- Interactive [dashboard](Churn%20Report.pbix) to monitor customer churn risk using Power BI.

---

## Metrics and Evaluation
- Accuracy: 0.95
- Precision (Class 0): 0.97
- Recall (Class 0): 0.97
- F1-Score (Class 0): 0.97
- Precision (Class 1): 0.81
- Recall (Class 1): 0.79
- F1-Score (Class 1): 0.80

- Confusion Matrix:
- TN: 19,767
- FP: 542
- FN: 579
- TP: 2,240

- ROC-AUC: 0.985
- PR-AUC: 0.903

---

## Dataset
Original dataset sourced from:
https://www.kaggle.com/datasets/sandiledesmondmfazi/bank-customer-churn
