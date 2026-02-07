# Heart Disease Prediction — ML Case Study (Leakage-Safe Pipeline)

A reproducible end-to-end machine learning workflow for predicting heart disease from tabular clinical features.  
This project focuses on correct data handling (including invalid values), leakage-safe preprocessing with scikit-learn Pipelines, model comparison, and final evaluation.

## Dataset
Source (Kaggle): https://www.kaggle.com/datasets/mexwell/heart-disease-dataset/data
Download and place the CSV file into this folder (e.g., `data/... .csv`) and update the path in the notebook if needed.

## About Dataset
This heart disease dataset is curated by combining 5 popular heart disease datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes.
The five datasets used for its curation are:
- Cleveland
- Hungarian
- Switzerland
- Long Beach VA
- Statlog (Heart) Data Set.

This dataset consists of 1190 instances with 11 features. These datasets were collected and combined at one place to help advance research on CAD-related machine learning and data mining algorithms, and hopefully to ultimately advance clinical diagnosis and early treatment.

## Features used (data dictionary)
The notebook groups features into numeric and categorical sets for preprocessing.

### Numeric features
- `age` - Age (years);
- `resting bp s` - Resting systolic blood pressure (mm Hg);
- `cholesterol` - Serum cholesterol (mg/dl);
- `max heart rate` - Maximum heart rate achieved;
- `oldpeak` - ST depression induced by exercise relative to rest.

### Categorical / nominal features (with codes)
- `chest pain type`:
  - 1 - Typical angina;
  - 2 - Atypical angina;
  - 3 - Non-anginal pain;
  - 4 - Asymptomatic;
- `fasting blood sugar` (fasting blood sugar > 120 mg/dl):
  - 1 - True;
  - 0 - False;
- `resting ecg`:
  - 0 - Normal;
  - 1 - ST-T wave abnormality (T wave inversions and/or ST elevation/depression > 0.05 mV);
  - 2 - Probable/definite left ventricular hypertrophy (Estes’ criteria);
- `ST slope`:
  - 1 - Upsloping;
  - 2 - Flat;
  - 3 - Downsloping;
- `sex`, `exercise angina` (binary indicators; see notebook for details).
- `target`:
  - 1 - heart disease;
  - 0 - no heart disease.
    
## Work done (project workflow)
- Exploratory Data Analysis (EDA): distributions, class balance, basic correlation analysis;
- Data quality checks and handling of invalid/impossible values;
- Leakage-safe preprocessing using `ColumnTransformer + Pipeline`:
  - Numeric: median imputation + standard scaling;
  - Categorical: most-frequent imputation + one-hot encoding;
- Model training and comparison:
  - Logistic Regression (baseline);
  - Random Forest;
  - XGBoost;
- Hyperparameter tuning via cross-validation (train only);
- Final evaluation on a held-out test set:
  - ROC-AUC and PR-AUC (threshold-independent ranking quality);
  - Precision/Recall/F1, confusion matrix, and classification report (threshold-dependent classification quality).

## Data quality handling (invalid values)
Some values are physiologically implausible and were treated as invalid measurements:
- `cholesterol = 0`;
- `resting bp s = 0`;
- `oldpeak < 0`.

These values are converted to `NaN` and handled inside the Pipeline.  
Doing this inside the Pipeline prevents data leakage: imputation/scaling/encoding are fitted only on training folds during CV.

## Results (test set)
Final comparison on a held-out test set:
- **Random Forest** achieved better ranking performance (higher ROC-AUC / PR-AUC);
- **XGBoost** achieved slightly better fixed-threshold classification (higher F1 and fewer mistakes in the confusion matrix);
- Logistic Regression provided a strong interpretable baseline.

**Metrics:**
- Logistic Regression: ROC-AUC = **0.9305**, PR-AUC = **0.9304**, F1 = **0.8819**  
- Random Forest: ROC-AUC = **0.9706**, PR-AUC = **0.9677**, F1 = **0.9319**  
- XGBoost: ROC-AUC = **0.9574**, PR-AUC = **0.9549**, F1 = **0.9396**

**Final choice:** XGBoost, because in this scenario I prioritize accurate healthy vs. diseased decisions at a fixed threshold (higher F1 and fewer errors).
