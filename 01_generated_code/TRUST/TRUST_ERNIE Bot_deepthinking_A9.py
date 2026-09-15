# -*- coding: utf-8 -*-
"""
随机森林分类模型构建与评估
适用于 PyCharm 2025.2.3 环境
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ===================== 1. 数据加载 =====================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征和目标
X = df.iloc[:, :-1]  # 所有特征列
y = df.iloc[:, -1]   # TRUST（最后一列）

# ===================== 2. 目标变量二分类转换 =====================
# TRUST >= 16 为1，否则为0
y_binary = (y >= 16).astype(int)

# ===================== 3. 定义变量类型 =====================
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal = ['TPPA']                     # 序数编码
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ===================== 4. 构建预处理管道 =====================
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量（独热编码）
categorical_onehot_transformer = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

# 分类变量（序数编码）
categorical_ordinal_transformer = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

# 列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal)
    ],
    remainder='drop'
)

# ===================== 5. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

# ===================== 6. 构建完整管道（含SMOTE） =====================
# 注意：SMOTE放在预处理之后，因为需要在数值特征空间中操作
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ===================== 7. 超参数调优 =====================
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
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

# ===================== 8. 输出最佳参数 =====================
print("=" * 50)
print("最佳超参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证 F1 分数: {grid_search.best_score_:.4f}")
print("=" * 50)

# ===================== 9. 模型评估 =====================
best_model = grid_search.best_estimator_

# 预测
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 50)
print("模型评估结果:")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):   {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("=" * 50)
