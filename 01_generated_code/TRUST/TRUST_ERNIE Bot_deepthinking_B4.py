# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 检验科TRUST滴度预测（二分类：TRUST >= 16）
适用于 PyCharm 2025.2.3 环境
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. 数据加载
# ============================================================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# ============================================================
# 2. 特征与目标变量定义
# ============================================================
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA',
                'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[feature_cols]
y = df['TRUST']  # 目标列（最后一列）

# 二分类目标：TRUST >= 16 为正类(1)，否则为负类(0)
y_binary = (y >= 16).astype(int)

# ============================================================
# 3. 列类型定义
# ============================================================
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal = ['TPPA']                       # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC',
                   'PLT', 'NC', 'LY', 'NLR']            # 连续变量（中位数填充）

# ============================================================
# 4. 构建预处理管道（ColumnTransformer）
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        # 独热编码（分类变量，无缺失值，无需填充）
        ('onehot', OneHotEncoder(drop='first', sparse_output=False),
         categorical_onehot),
        # 序数编码（分类变量，无缺失值）
        ('ordinal', OrdinalEncoder(),
         categorical_ordinal),
        # 连续变量：中位数填充
        ('continuous', SimpleImputer(strategy='median'),
         continuous_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ============================================================
# 5. 构建完整Pipeline（预处理 + SMOTE + 随机森林）
# ============================================================
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ============================================================
# 6. 划分训练集与测试集
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary  # 保持类别比例
)

# ============================================================
# 7. 超参数网格与GridSearchCV调优
# ============================================================
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
    scoring='roc_auc',
    n_jobs=1,          # 不使用多进程
    verbose=1,
    refit=True
)

# ============================================================
# 8. 模型训练
# ============================================================
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("=" * 60)
print("最佳超参数组合：")
print(grid_search.best_params_)
print("=" * 60)

# ============================================================
# 9. 模型预测
# ============================================================
best_model = grid_search.best_estimator_

# 预测类别
y_pred = best_model.predict(X_test)

# 预测概率（用于计算AUC，取正类概率）
y_prob = best_model.predict_proba(X_test)[:, 1]

# ============================================================
# 10. 模型评估
# ============================================================
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("模型评估结果（测试集）：")
print(f"  准确率 (Accuracy)  : {accuracy:.4f}")
