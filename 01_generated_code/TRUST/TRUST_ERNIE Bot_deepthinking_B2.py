# -*- coding: utf-8 -*-
"""
随机森林分类模型 - TRUST滴度≥16预测
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

# ========================== 1. 数据加载 ==========================
print("=" * 50)
print("步骤1：加载数据")
print("=" * 50)

df = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征和目标
X = df.iloc[:, :-1]  # 所有特征列
y = df.iloc[:, -1]   # 最后一列 TRUST

print(f"数据集形状: {df.shape}")
print(f"特征列: {list(X.columns)}")
print(f"目标列: {y.name}")
print(f"目标列唯一值: {y.unique()}")

# ========================== 2. 目标变量转换（二分类） ==========================
print("\n" + "=" * 50)
print("步骤2：目标变量二分类转换（TRUST ≥ 16）")
print("=" * 50)

y_binary = (y >= 16).astype(int)
print(f"转换后类别分布:\n{y_binary.value_counts()}")
print(f"正类（≥16）比例: {y_binary.mean():.4f}")

# ========================== 3. 定义变量类型 ==========================
print("\n" + "=" * 50)
print("步骤3：定义变量类型")
print("=" * 50)

categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal = ['TPPA']                      # 序数编码
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

print(f"独热编码变量: {categorical_onehot}")
print(f"序数编码变量: {categorical_ordinal}")
print(f"连续变量: {continuous}")

# ========================== 4. 数据集划分 ==========================
print("\n" + "=" * 50)
print("步骤4：划分训练集和测试集")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ========================== 5. 构建预处理管道 ==========================
print("\n" + "=" * 50)
print("步骤5：构建预处理管道")
print("=" * 50)

# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量：独热编码（无缺失值，不需要填充）
categorical_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# TPPA：序数编码（无缺失值，不需要填充）
categorical_ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 组合所有预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal)
    ],
    remainder='drop'  # 不在列表中的列丢弃
)

# ========================== 6. 构建完整模型管道（含SMOTE） ==========================
print("\n" + "=" * 50)
print("步骤6：构建含SMOTE的完整管道")
print("=" * 50)

# 随机森林分类器（初始参数）
rf_classifier = RandomForestClassifier(
    random_state=42,
    n_jobs=1  # 不使用多进程
)

# 完整管道：预处理 -> SMOTE -> 随机森林
pipeline = ImbPipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', rf_classifier)
    ]
)

# ========================== 7. 超参数调优 ==========================
print("\n" + "=" * 50)
print("步骤7：超参数调优（GridSearchCV）")
print("=" * 50)

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
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 获取最佳模型
best_model = grid_search.best_estimator_

# ========================== 8. 模型评估 ==========================
print("\n" + "=" * 50)
print("步骤8：模型评估")
print("=" * 50)

# 预测
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*40}")
print(f"模型评估结果（测试集）")
print(f"{'='*40}")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print(f"{'='*40}")

# 额外输出：测试集预测分布
print(f"\n测试集实际分布:\n{y_test.value_counts().sort_index()}")
print(f"测试集预测分布:\n{pd.Series(y_pred).value_counts().sort_index()}")

print("\n" + "=" * 50)
print("模型构建与评估完成！")
print("=" * 50)
