# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 构建与评估
适用于 PyCharm 2025.2.3 环境
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ======================== 1. 加载数据 ========================
print("=" * 60)
print("步骤1: 加载数据")
print("=" * 60)

df = pd.read_csv('train_data.csv', encoding='utf-8')

print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# ======================== 2. 分离特征与目标 ========================
print("\n" + "=" * 60)
print("步骤2: 分离特征与目标变量")
print("=" * 60)

feature_columns = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                   'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[feature_columns].copy()
y = df['TRUST'].copy()  # 最后一列为目标列

# 创建二分类目标: TRUST >= 16 为1，否则为0
y_binary = (y >= 16).astype(int)
print(f"原始目标列类别分布:\n{y.value_counts().sort_index()}")
print(f"二分类目标列类别分布:\n{y_binary.value_counts()}")

# ======================== 3. 定义变量类型 ========================
print("\n" + "=" * 60)
print("步骤3: 定义变量类型")
print("=" * 60)

categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal_cols = ['TPPA']                      # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

print(f"独热编码列: {categorical_onehot_cols}")
print(f"序数编码列: {categorical_ordinal_cols}")
print(f"连续变量列: {continuous_cols}")

# ======================== 4. 构建预处理管道 ========================
print("\n" + "=" * 60)
print("步骤4: 构建预处理管道")
print("=" * 60)

# 连续变量: 中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量(独热编码): 无缺失值，直接编码
categorical_onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# 分类变量(序数编码): 无缺失值，直接编码
categorical_ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

# 组合所有预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot_cols),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'  # 不包含的列丢弃
)

print("预处理管道构建完成")

# ======================== 5. 划分训练集和测试集 ========================
print("\n" + "=" * 60)
print("步骤5: 划分训练集与测试集")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"训练集正类比例: {y_train.mean():.4f}")
print(f"测试集正类比例: {y_test.mean():.4f}")

# ======================== 6. 预处理 + SMOTE + 模型构建 ========================
print("\n" + "=" * 60)
print("步骤6: 数据预处理 + SMOTE + 模型构建")
print("=" * 60)

# 先对训练集进行预处理
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"预处理后训练集特征维度: {X_train_processed.shape}")
print(f"预处理后测试集特征维度: {X_test_processed.shape}")

# 使用SMOTE处理分类不平衡
print("\n应用SMOTE进行过采样...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print(f"SMOTE后训练集大小: {X_train_smote.shape[0]}")
print(f"SMOTE后正类数量: {y_train_smote.sum()}, 负类数量: {(y_train_smote == 0).sum()}")

# ======================== 7. 超参数调优 (GridSearchCV) ========================
print("\n" + "=" * 60)
print("步骤7: 超参数调优 (GridSearchCV)")
print("=" * 60)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=1)

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

grid_search.fit(X_train_smote, y_train_smote)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 获取最佳模型
best_model = grid_search.best_estimator_

# ======================== 8. 模型评估 ========================
print("\n" + "=" * 60)
print("步骤8: 模型评估")
print("=" * 60)

# 在测试集上进行预测
y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n模型评估结果:")
print(f"  准确率 (Accuracy):  {accuracy:.4f}")
print(f"  召回率 (Recall):    {recall:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  F1分数 (F1-score):  {f1:.4f}")
print(f"  AUC:                {auc:.4f}")

print("\n" + "=" * 60)
print("模型构建与评估完成!")
print("=" * 60)
