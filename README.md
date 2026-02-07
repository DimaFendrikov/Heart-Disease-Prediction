## Problem statement: why this work exists and what it is useful for

Predicting heart disease from basic clinical measurements is a typical real-world classification task on tabular data: there is a binary target, mixed feature types (numeric + categorical), and the results are evaluated not only by “accuracy”, but by metrics that reflect ranking quality and decision quality at a fixed threshold.

However, the main difficulty in such tasks is often not the choice of a model, but **the correctness of the workflow**. In practice, tabular medical-like datasets contain measurement issues: missing values, invalid values, inconsistent coding, and artifacts that can silently distort evaluation. A common mistake in ML projects is to “clean and transform everything first” and only then run cross-validation. This can lead to **data leakage**, where preprocessing learns information from the full dataset (including validation folds) and inflates metrics.

This project is useful as a **clean, reproducible reference** for how to do a tabular classification pipeline correctly:
- how to detect and handle invalid/impossible values (e.g., `cholesterol = 0`, `resting bp s = 0`, `oldpeak < 0`);
- how to compare different data-quality handling strategies in a measurable way, not “by intuition”;
- how to build an end-to-end **leakage-safe** preprocessing and modeling workflow using `ColumnTransformer + Pipeline`;
- how to compare models under stratified cross-validation and then validate the final choice on a held-out test set.

In other words, the practical value of the work is not that it “solves medicine”, but that it demonstrates solid engineering thinking in ML:
the model is evaluated honestly, preprocessing is not leaking information, and the approach is structured so that it can be reused for other tabular classification problems (finance, operations, risk scoring, churn, etc.) where data quality issues are common.

### What exactly is studied in this notebook
The notebook focuses on one core question:
**How do data-quality decisions affect model performance and reliability?**

To answer it, I compare multiple strategies for handling invalid measurements:
- dropping corrupted rows (`DROP`);
- converting invalid values to `NaN` and imputing (`IMPUTE_MEDIAN`);
- imputing while also adding indicator flags (`IMPUTE+FLAGS`), so the model can learn that a value was missing/invalid.

This is evaluated consistently using the same leakage-safe pipeline and the same CV protocol across models (Logistic Regression, Random Forest, XGBoost), and then validated on a separate test set.

### Important note about scope
This is an educational ML case study.  
The goal is to demonstrate a correct and reproducible ML workflow, not to provide a clinical diagnostic tool.
