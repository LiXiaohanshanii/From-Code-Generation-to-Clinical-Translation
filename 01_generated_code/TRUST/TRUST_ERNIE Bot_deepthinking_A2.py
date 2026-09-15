# -*- coding: utf-8 -*-
"""
随机森林分类模型构建与评估
适用环境：PyCharm 2025.2.3
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
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score, roc_auc_score)

# ============================================================
# 1. 数据加载
# ============================================================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# ============================================================
# 2. 目标变量构建（二分类：TRUST >= 16 为1，否则为0）
# ============================================================
df['TARGET'] = (df['TRUST'] >= 16).astype(int)

# 分离特征与目标
feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
X = df[feature_cols]
y = df['TARGET']

# ============================================================
# 3. 划分训练集与测试集
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 4. 定义列类型
# ============================================================
categorical_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_ordinal = ['TPPA']                   # 序数编码
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC',
              'PLT', 'NC', 'LY', 'NLR']         # 连续变量

# ============================================================
# 5. 构建预处理管道
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first', sparse_output=False), categorical_ohe),
        ('ordinal', OrdinalEncoder(), categorical_ordinal),
        ('continuous', SimpleImputer(strategy='median'), continuous)
    ],
    remainder='drop'
)

# ============================================================
# 6. 构建完整管道（预处理 + SMOTE + 随机森林）
# ============================================================
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ============================================================
# 7. 超参数网格
# ============================================================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# ============================================================
# 8. 网格搜索（不使用多进程）
# ============================================================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ============================================================
# 9. 获取最优模型并预测
# ============================================================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# ============================================================
# 10. 模型评估
# ============================================================
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=" * 60)
print("随机森林分类模型评估结果")
print("=" * 60)
print(f"最优超参数组合:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print("-" * 60)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("=" * 60)
