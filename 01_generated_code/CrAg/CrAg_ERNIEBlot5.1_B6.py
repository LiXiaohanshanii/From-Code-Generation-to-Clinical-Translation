# -*- coding: utf-8 -*-
"""
随机森林分类模型 - CSF-T预测 (≥20为正类)
适用于 PyCharm 2025.2.3 环境
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

# ========================= 1. 数据加载 =========================
data = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ========================= 2. 特征与目标分离 =========================
feature_columns = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                   'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']

X = data[feature_columns].copy()
y = data.iloc[:, -1].copy()  # CSF-T（最后一列）

# ========================= 3. 目标变量二分类处理 =========================
y_binary = (y >= 20).astype(int)  # ≥20为1，<20为0

# ========================= 4. 定义变量类型 =========================
# 分类变量（独热编码）
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
onehot_cols = [c for c in onehot_cols if c in X.columns]

# 分类变量（序数编码）
ordinal_cols = ['SER-T', 'Ink staining']
ordinal_cols = [c for c in ordinal_cols if c in X.columns]

# 连续变量
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
continuous_cols = [c for c in continuous_cols if c in X.columns]

# ========================= 5. 预处理管道构建 =========================

# --- 连续变量：中位数填充 ---
continuous_transformer = SimpleImputer(strategy='median')

# --- 序数编码 ---
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# --- 独热编码 ---
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# --- 列转换器 ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('ohe', onehot_transformer, onehot_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ========================= 6. 划分训练集和测试集 =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ========================= 7. 构建完整Pipeline（含SMOTE + 随机森林） =========================
# 定义SMOTE（仅在训练阶段使用）
smote = SMOTE(random_state=42)

# 定义随机森林分类器
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=1  # 不使用多进程
)

# 完整Pipeline
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', rf_clf)
])

# ========================= 8. 超参数调优 =========================
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
    verbose=1
)

# 拟合模型
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("=" * 60)
print("最佳超参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
print("=" * 60)

# ========================= 9. 模型评估 =========================
# 使用最佳模型预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("模型评估结果（测试集）:")
print(f"准确率  (Accuracy):  {accuracy:.4f}")
print(f"召回率  (Recall):    {recall:.4f}")
print(f"精确率  (Precision): {precision:.4f}")
print(f"F1分数  (F1-score):  {f1:.4f}")
print(f"AUC:                 {auc:.4f}")
print("=" * 60)