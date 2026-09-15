# -*- coding: utf-8 -*-
"""
脑脊液隐球菌抗原(CrAg)滴度预测模型 - 随机森林二分类(含数据集划分)
适用环境: PyCharm 2025.2.3 / Python 3.9+
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, classification_report)

# =============================================================================
# 1. 数据加载与目标变量构建
# =============================================================================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'

X = df[feature_cols].copy()
y = (df[target_col] >= 20).astype(int)  # 二分类: ≥20为1, <20为0

# =============================================================================
# 2. 【关键】分层划分训练集与测试集
# =============================================================================
# test_size=0.2: 80%训练, 20%测试
# stratify=y: 分层抽样, 保证正负样本比例在两个子集中一致
# random_state=42: 固定随机种子, 确保结果可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"原始数据集大小: {X.shape[0]}")
print(f"训练集大小: {X_train.shape[0]}, 阳性率: {y_train.mean():.3f}")
print(f"测试集大小: {X_test.shape[0]}, 阳性率: {y_test.mean():.3f}")

# =============================================================================
# 3. 定义预处理管道 (与之前相同)
# =============================================================================
continuous_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
ordinal_features = ['SER-T', 'Ink staining']
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', Pipeline([('imp', SimpleImputer(strategy='median'))]), continuous_features),
        ('ord', Pipeline([('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), ordinal_features),
        ('cat', Pipeline([('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), onehot_features)
    ],
    remainder='drop'
)

# =============================================================================
# 4. 构建Pipeline (SMOTE仅作用于训练集)
# =============================================================================
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),      # ✅ 仅在fit(X_train)时生效
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# =============================================================================
# 5. 超参数调优 (仅在训练集上进行)
# =============================================================================
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1,       # 不使用多进程
    verbose=1,
    refit=True
)

print("\n开始在【训练集】上进行超参数搜索...")
grid_search.fit(X_train, y_train)  # ✅ 只用训练集

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# =============================================================================
# 6. 【关键】在独立测试集上评估模型
# =============================================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)       # ✅ 测试集不参与任何预处理拟合
y_prob = best_model.predict_proba(X_test)[:, 1]

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_prob)
}

print("\n" + "="*50)
print("独立测试集评估结果")
print("="*50)
for name, value in metrics.items():
    print(f"{name:<12}: {value:.4f}")
print("="*50)
print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=['阴性(<20)', '阳性(≥20)']))

# 保存模型
import joblib
joblib.dump(best_model, 'crag_rf_best_model.pkl')
print("最佳模型已保存为: crag_rf_best_model.pkl")