# -*- coding: utf-8 -*-
"""
随机森林分类模型构建与评估
适用于 PyCharm 2025.2.3 环境
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==================== 1. 数据加载 ====================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# ==================== 2. 特征与目标列定义 ====================
feature_columns = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                   'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
target_column = 'TRUST'  # 最后一列

X = data[feature_columns]
y = data[target_column]

# ==================== 3. 目标变量二分类转换 ====================
# 预测是否 >= 16
y_binary = (y >= 16).astype(int)

print(f"原始目标分布:\n{y.value_counts().sort_index()}")
print(f"二分类目标分布:\n{y_binary.value_counts()}")

# ==================== 4. 定义变量类型 ====================
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal = ['TPPA']                      # 序数编码
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ==================== 5. 构建预处理管道 ====================
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量（独热编码）
categorical_onehot_transformer = OneHotEncoder(drop='first', sparse_output=False)

# 分类变量（序数编码）
categorical_ordinal_transformer = OrdinalEncoder()

# 组合所有预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal)
    ],
    remainder='drop'
)

# ==================== 6. 划分训练集与测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\n训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# ==================== 7. 构建完整Pipeline（含SMOTE） ====================
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
])

# ==================== 8. 超参数调优 ====================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    verbose=1
)

print("\n开始超参数调优...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ==================== 9. 使用最佳模型预测 ====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# ==================== 10. 模型评估 ====================
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 50)
print("模型评估结果（测试集）")
print("=" * 50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("=" * 50)
