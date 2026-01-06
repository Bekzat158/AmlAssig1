
# Assignment 1 Report

## Dataset
**Name**: Medical Cost Personal Datasets (Insurance)  
**Source**: [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) or [Github Raw](https://raw.githubusercontent.com/mirichoi0218/insurance/master/insurance.csv)  
**Description**: The dataset includes 1338 examples of beneficiaries with features: age, sex, bmi, children, smoker, region, and charges (target).

## Source Code
The source code is available in `assignment1.py` within this directory.
It uses `scikit-learn`, `pandas`, `xgboost`, `lightgbm`, and `catboost` to train 11 regression models using 10-fold cross-validation.  
- Categorical features (`sex`, `smoker`, `region`) are One-Hot Encoded.
- Numerical features (`age`, `bmi`, `children`) are Scaled (StandardScalar).
- **Note**: The original task is Regression. To generate Classification metrics (Accuracy, Precision, Recall, F1), the target `charges` variable was binned into 3 classes (Low, Medium, High) using quantiles, and equivalent Classifier algorithms were trained alongside the Regressors.

## Results (Table 1)

| Algorithm | Number of features | Number of targets | k-fold validation | Accuracy_score/Precision/Recall/F1 score<br>(for classification tasks) | RMSE/R²<br>(for regression tasks) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ridge | 6 | 1 | 10 | 0.8438 / 0.8727 / 0.8438 / 0.8370 | 6070.39 / 0.7391 |
| Lasso | 6 | 1 | 10 | 0.8498 / 0.8807 / 0.8498 / 0.8432 | 6070.27 / 0.7391 |
| Elastic Net | 6 | 1 | 10 | 0.8415 / 0.8767 / 0.8415 / 0.8361 | 9528.57 / 0.3726 |
| KNN (Regression/Classification) | 6 | 1 | 10 | 0.8288 / 0.8371 / 0.8288 / 0.8257 | 6488.81 / 0.7045 |
| Extra Trees | 6 | 1 | 10 | 0.8759 / 0.8806 / 0.8759 / 0.8753 | 5204.29 / 0.8081 |
| Adaptive Boosting | 6 | 1 | 10 | 0.8042 / 0.8132 / 0.8042 / 0.8005 | 5102.55 / 0.8165 |
| Gradient Boosting | 6 | 1 | 10 | 0.8961 / 0.9028 / 0.8961 / 0.8949 | 4517.51 / 0.8539 |
| High-Performance Boosting (XGBoost) | 6 | 1 | 10 | 0.8804 / 0.8837 / 0.8804 / 0.8796 | 5157.01 / 0.8099 |
| High-Performance Boosting (LightGBM) | 6 | 1 | 10 | 0.8819 / 0.8858 / 0.8819 / 0.8809 | 4796.73 / 0.8365 |
| CatBoost | 6 | 1 | 10 | 0.8946 / 0.9004 / 0.8946 / 0.8939 | 4719.99 / 0.8412 |
| HistGradientBoosting | 6 | 1 | 10 | 0.8804 / 0.8844 / 0.8804 / 0.8796 | 4778.87 / 0.8372 |

**Note**: "Number of features" refers to the original input columns (age, sex, bmi, children, smoker, region) before encoding. The processed One-Hot Encoded feature space is slightly larger.

## Brief Analysis

1.  **Best Performance**: **Gradient Boosting** achieved the best results across both tasks.
    - **Regression**: Lowest RMSE (**4517.51**) and highest R² (**0.8539**).
    - **Classification**: Highest Accuracy (**89.61%**) and F1 Score (**0.8949**).
    CatBoost also performed exceptionally well, closely following Gradient Boosting.
2.  **Boosting Power**: Tree-based boosting methods (Gradient Boosting, CatBoost, LightGBM, XGBoost) consistently outperformed linear models and KNN. This confirms strong non-linear patterns in the data (likely smoker/bmi interactions).
3.  **Linear Models**: Ridge and Lasso showed decent Regression performance (R² ~0.74) and surprisingly good Classification accuracy (~84-85%), likely because separating "Low", "Medium", and "High" charge groups is partially linear, even if the exact charge prediction is not. Elastic Net performed poorly in Regression (R² ~0.37) but recovered in Classification (Acc ~84%), performing similarly to other linear classifiers.
4.  **Conclusion**: Gradient Boosting is the superior choice for this dataset. The high classification scores (>89%) indicate that identifying the cost tier (Low/Medium/High) is a highly solvable problem with these features, even if exact regression prediction has some residual error.
