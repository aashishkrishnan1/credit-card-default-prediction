# Predicting Credit Card Default: A Machine Learning Approach

A binary classification project using machine learning to predict whether a credit card client will default on their next payment. Built out of an interest in how quantitative modeling can be applied to real-world problems in consumer finance and credit risk.

## Overview

Credit default prediction sits at the intersection of finance and machine learning — lenders use models like these to inform underwriting decisions, set credit limits, and estimate loss reserves. This project explores that problem using a real dataset of 30,000 credit card holders in Taiwan, comparing four classification models to identify which best predicts next-month default.

## Dataset

**UCI Default of Credit Card Clients**  
30,000 observations | 23 features | Binary outcome (default: yes/no)  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

Features include credit limit, demographic information (age, sex, education, marital status), and six months of payment history, bill amounts, and payment amounts (April–September 2005).

## Models Compared

| Model | CV ROC-AUC |
|---|---|
| Gradient Boosted Trees | 0.7836 |
| Random Forest | 0.7708 |
| Logistic Regression | 0.7263 |
| Elastic Net | 0.7262 |

All models tuned with 5-fold stratified cross-validation and grid search. ROC-AUC was chosen as the primary metric because the dataset is class-imbalanced (~22% default rate), making raw accuracy misleading.

## Results

The best model (Gradient Boosted Trees) achieved a **test ROC-AUC of 0.7764** and **82% accuracy** on the held-out test set — outperforming the benchmark established in the original published paper on this dataset (~0.72 AUC).

Tree-based ensemble methods substantially outperformed linear models, suggesting nonlinear relationships and interaction effects in the data that logistic regression cannot capture without manual feature engineering.

**Most predictive features:** Recent payment status variables (especially September repayment status) were by far the strongest predictors of default — more so than demographic features or bill amounts.

## Key Steps

- Exploratory data analysis: class distribution, credit limit distributions, default rates by education level, payment history correlations
- Data cleaning: removal of undocumented categorical codes, clipping of anomalous payment values
- Feature engineering: one-hot encoding of categorical variables, standard scaling for linear models
- Hyperparameter tuning: grid search with stratified 5-fold CV for elastic net, random forest, and gradient boosted trees
- Evaluation: ROC curves, confusion matrices, classification reports, permutation feature importance

## Tools & Libraries

Python | pandas | scikit-learn | matplotlib | seaborn | NumPy

## Limitations & Future Work

- Dataset reflects a single credit market (Taiwan, 2005) and may not generalize broadly
- Decision threshold was not optimized for real-world cost asymmetry between false positives and false negatives
- Future directions: fairness-aware modeling to audit for disparate impact, incorporating macroeconomic covariates, exploring XGBoost or LightGBM for improved performance
