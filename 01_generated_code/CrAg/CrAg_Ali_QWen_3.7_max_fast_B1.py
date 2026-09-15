# -*- coding: utf-8 -*-
"""
CSF CrAg 滴度二分类预测模型 (Train/Test Split + SMOTE)
适用环境: PyCharm 2025.2.3 / Python 3.9+
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

warnings.filterwarnings('ignore')

# ===================== 1. 数据加载与标签构建 =====================
print(">>> 正在加载数据...")
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'

X = df[feature_cols].copy()
y = (df[target_col] >= 20).astype(int)

# ===================== 2. ⭐ 划分训练集与测试集 =====================
# stratify=y 确保训练集和测试集中阳性(≥20)的比例一致
# random_state=42 保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]} | 阳性比例: {y_train.mean():.4f}")
print(f"测试集大小: {X_test.shape[0]} | 阳性比例: {y_test.mean():.4f}")

# ===================== 3. 定义预处理与Pipeline =====================
ordinal_features = ['SER-T', 'Ink staining']
ohe_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
continuous_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_features),
        ('num', SimpleImputer(strategy='median'), continuous_features)
    ],
    remainder='drop'
)

# SMOTE放在Pipeline内部 → 仅在GridSearchCV的训练折中执行，不会污染验证折和最终测试集
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ===================== 4. 超参数调优 (仅在训练集上) =====================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=1,
    verbose=1,
    refit=True
)

print("\n>>> 在训练集上进行超参数搜索...")
grid_search.fit(X_train, y_train)  # ⭐ 仅用训练集拟合

print(f"最佳参数: {grid_search.best_params_}")
print(f"训练集交叉验证 F1: {grid_search.best_score_:.4f}")

# ===================== 5. ⭐ 在独立测试集上评估 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)      # 测试集未经SMOTE处理
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("     独立测试集评估结果 (真实泛化性能)")
print("=" * 50)
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  AUC      : {roc_auc_score(y_test, y_prob):.4f}")
print("=" * 50)

joblib.dump(best_model, 'crag_rf_best_model.pkl')
print("\n>>> 模型已保存至 crag_rf_best_model.pkl")