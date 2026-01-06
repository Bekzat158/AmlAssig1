import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier, 
                              AdaBoostRegressor, AdaBoostClassifier, 
                              GradientBoostingRegressor, GradientBoostingClassifier, 
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# 1. Load Dataset
data = pd.read_csv('insurance.csv')

# 2. Preprocessing
target_col = 'charges'
X = data.drop(columns=[target_col])
y_reg = data[target_col]

# Create Classification Target (Binning into 3 classes: Low, Medium, High)
# Using quantiles to ensure balanced classes
y_clf_raw = pd.qcut(data[target_col], q=3, labels=['Low', 'Medium', 'High'])
le = LabelEncoder()
y_clf = le.fit_transform(y_clf_raw)

categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# 3. Define Algorithms (Pairs of Regressor, Classifier)
# Note: For Lasso/ElasticNet, LogisticRegression is the closest classification equivalent with penalties.
# We map the concept "Lasso" to LogisticRegression(penalty='l1') and "Elastic Net" to LogisticRegression(penalty='elasticnet')
models = [
    {
        "name": "Ridge", 
        "reg": Ridge(), 
        "clf": RidgeClassifier()
    },
    {
        "name": "Lasso", 
        "reg": Lasso(), 
        "clf": LogisticRegression(penalty='l1', solver='liblinear')
    },
    {
        "name": "Elastic Net", 
        "reg": ElasticNet(), 
        "clf": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
    },
    {
        "name": "KNN", # Renamed from "KNN Regression" to be generic
        "reg": KNeighborsRegressor(), 
        "clf": KNeighborsClassifier()
    },
    {
        "name": "Extra Trees", # Renamed to be generic
        "reg": ExtraTreesRegressor(), 
        "clf": ExtraTreesClassifier()
    },
    {
        "name": "Adaptive Boosting", 
        "reg": AdaBoostRegressor(), 
        "clf": AdaBoostClassifier()
    },
    {
        "name": "Gradient Boosting", 
        "reg": GradientBoostingRegressor(), 
        "clf": GradientBoostingClassifier()
    },
    {
        "name": "XGBoost", 
        "reg": XGBRegressor(verbosity=0), 
        "clf": XGBClassifier(verbosity=0, use_label_encoder=False, eval_metric='logloss')
    },
    {
        "name": "LightGBM", 
        "reg": LGBMRegressor(verbosity=-1), 
        "clf": LGBMClassifier(verbosity=-1)
    },
    {
        "name": "CatBoost", 
        "reg": CatBoostRegressor(verbose=0), 
        "clf": CatBoostClassifier(verbose=0)
    },
    {
        "name": "HistGradientBoosting", 
        "reg": HistGradientBoostingRegressor(), 
        "clf": HistGradientBoostingClassifier()
    }
]

# 4. K-Fold Validation
cv_reg = KFold(n_splits=10, shuffle=True, random_state=42)
cv_clf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = []

# Header for the combined table
print(f"{'Algorithm':<30} | {'RMSE':<12} | {'R2':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
print("-" * 110)

for item in models:
    name = item['name']
    
    # --- Regression Task ---
    pipe_reg = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', item['reg'])])
    scoring_reg = {'neg_root_mean_squared_error': 'neg_root_mean_squared_error', 'r2': 'r2'}
    scores_reg = cross_validate(pipe_reg, X, y_reg, cv=cv_reg, scoring=scoring_reg)
    
    rmse = -scores_reg['test_neg_root_mean_squared_error'].mean()
    r2 = scores_reg['test_r2'].mean()

    # --- Classification Task ---
    pipe_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', item['clf'])])
    # 'weighted' average handles multi-class (Low, Medium, High) nicely
    scoring_clf = {
        'accuracy': 'accuracy', 
        'precision': 'precision_weighted', 
        'recall': 'recall_weighted', 
        'f1': 'f1_weighted'
    }
    scores_clf = cross_validate(pipe_clf, X, y_clf, cv=cv_clf, scoring=scoring_clf)
    
    accuracy = scores_clf['test_accuracy'].mean()
    precision = scores_clf['test_precision'].mean()
    recall = scores_clf['test_recall'].mean()
    f1 = scores_clf['test_f1'].mean()

    print(f"{name:<30} | {rmse:<12.4f} | {r2:<8.4f} | {accuracy:<10.4f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")
    
    results.append({
        "Algorithm": name,
        "RMSE": rmse,
        "R2": r2,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
