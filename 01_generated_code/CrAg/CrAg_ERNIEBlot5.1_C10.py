# -*- coding: utf-8 -*-
"""
随机森林分类模型 - CrAg预测（CSF-T是否≥20）
环境: PyCharm 2025.2.3
注意: 使用imblearn.pipeline.Pipeline防止数据泄露
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ======================== 1. 数据加载 ========================
print("=" * 60)
print("步骤1: 加载数据")
print("=" * 60)
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# ======================== 2. 特征与目标分离 ========================
# 特征列（前12列）
X = df.iloc[:, :-1]
# 目标列（最后一列 CSF-T）
y_raw = df.iloc[:, -1]

# 二分类目标: ≥20 为1, <20 为0
y = (y_raw >= 20).astype(int)
print(f"\n目标变量分布:\n{y.value_counts()}")

# ======================== 3. 定义变量类型 ========================
# 连续变量
continuous_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 独热编码变量
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

# 序数编码变量
ordinal_features = ['SER-T', 'Ink staining']

# ======================== 4. 划分训练集和验证集 ========================
print("\n" + "=" * 60)
print("步骤2: 划分训练集和验证集 (4:1)")
print("=" * 60)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集样本数: {X_train.shape[0]}")
print(f"验证集样本数: {X_val.shape[0]}")
print(f"训练集类别分布:\n{y_train.value_counts()}")
print(f"验证集类别分布:\n{y_val.value_counts()}")

# ======================== 5. 构建预处理流水线 ========================
print("\n" + "=" * 60)
print("步骤3: 构建预处理流水线 (数值→独热→序数)")
print("=" * 60)

# 5.1 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 5.2 独热编码变量
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 5.3 序数编码变量
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 5.4 使用ColumnTransformer组合（按顺序: 数值→独热→序数）
preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', continuous_transformer, continuous_features),
        ('onehot', onehot_transformer, onehot_features),
        ('ordinal', ordinal_transformer, ordinal_features),
    ],
    remainder='drop'
)

# ======================== 6. 构建完整Pipeline（含SMOTE） ========================
print("\n" + "=" * 60)
print("步骤4: 构建含SMOTE的完整Pipeline")
print("=" * 60)

# 随机森林分类器（初始参数）
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)

# 使用imblearn.pipeline.Pipeline将SMOTE嵌入交叉验证内部
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_classifier)
])

# ======================== 7. 超参数调优 ========================
print("\n" + "=" * 60)
print("步骤5: 超参数调优 (GridSearchCV, n_jobs=1)")
print("=" * 60)

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 分层5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

print(f"\n最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 获取最佳模型
best_model = grid_search.best_estimator_

# ======================== 8. 模型评估 ========================
print("\n" + "=" * 60)
print("步骤6: 在验证集上评估模型")
print("=" * 60)

# 预测
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print("\n模型评估结果:")
print(f"  准确率 (Accuracy):  {accuracy:.4f}")
print(f"  召回率 (Recall):    {recall:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  F1分数 (F1-score):  {f1:.4f}")
print(f"  AUC:                {auc:.4f}")

print("\n" + "=" * 60)
print("模型构建与评估完成！")
print("=" * 60)