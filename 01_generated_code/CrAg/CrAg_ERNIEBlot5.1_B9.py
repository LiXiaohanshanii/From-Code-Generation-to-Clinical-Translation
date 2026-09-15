# -*- coding: utf-8 -*-
"""
随机森林分类模型 - CrAg预测
目标：预测CSF-T是否≥20（二分类任务）
环境：PyCharm 2025.2.3
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# ===================== 1. 数据加载 =====================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ===================== 2. 特征与目标分离 =====================
# 目标列：CSF-T（最后一列）
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 二分类标签：≥20 为1，<20 为0
y = (y >= 20).astype(int)

# ===================== 3. 分类变量定义 =====================
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_cols = ['SER-T', 'Ink staining']
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# ===================== 4. 构建预处理管道 =====================
# 连续变量缺失值：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 序数编码
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

# 独热编码
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('onehot', onehot_transformer, onehot_cols)
    ],
    remainder='drop'
)

# ===================== 5. 构建完整建模管道（含SMOTE） =====================
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ===================== 6. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 7. 超参数调优（GridSearchCV） =====================
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
    n_jobs=1,  # 不使用多进程
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 最佳模型
best_model = grid_search.best_estimator_

# ===================== 8. 模型评估 =====================
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# ===================== 9. 输出结果 =====================
print("=" * 60)
print("随机森林分类模型评估结果")
print("=" * 60)
print(f"最佳超参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
print("-" * 60)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("=" * 60)