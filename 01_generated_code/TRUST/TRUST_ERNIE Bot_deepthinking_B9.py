# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 预测TRUST是否≥16
适用于PyCharm 2025.2.3环境
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# ===================== 1. 加载数据 =====================
print("=" * 60)
print("步骤1: 加载数据")
print("=" * 60)

df = pd.read_csv('train_data.csv', encoding='utf-8')

print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# ===================== 2. 定义特征与目标 =====================
# 特征列
feature_columns = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                   'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 目标列（最后一列）
target_column = df.columns[-1]  # TRUST

X = df[feature_columns].copy()
y = df[target_column].copy()

# 二分类转化：TRUST ≥ 16 为1，否则为0
y_binary = (y >= 16).astype(int)
print(f"\n目标变量分布:\n{y_binary.value_counts()}")
print(f"  0 (TRUST < 16): {(y_binary == 0).sum()}")
print(f"  1 (TRUST ≥ 16): {(y_binary == 1).sum()}")

# ===================== 3. 定义变量类型 =====================
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal = ['TPPA']                      # 序数编码
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

print(f"\n独热编码变量: {categorical_onehot}")
print(f"序数编码变量: {categorical_ordinal}")
print(f"连续变量: {continuous}")

# ===================== 4. 构建预处理管道 =====================
print("\n" + "=" * 60)
print("步骤2: 构建预处理管道")
print("=" * 60)

# 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量（独热编码）
categorical_onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# 分类变量（序数编码）
categorical_ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

# 组合所有预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous),
        ('cat_ohe', categorical_onehot_transformer, categorical_onehot),
        ('cat_ord', categorical_ordinal_transformer, categorical_ordinal)
    ],
    remainder='drop'
)

# ===================== 5. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ===================== 6. 对训练集进行预处理和SMOTE =====================
print("\n" + "=" * 60)
print("步骤3: 数据预处理 + SMOTE过采样")
print("=" * 60)

# 先对训练集进行预处理
X_train_processed = preprocessor.fit_transform(X_train)

print(f"预处理后训练集特征维度: {X_train_processed.shape}")

# SMOTE过采样处理类别不平衡
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print(f"SMOTE后训练集大小: {X_train_smote.shape[0]}")
print(f"SMOTE后类别分布:\n{pd.Series(y_train_smote).value_counts()}")

# 对测试集仅进行预处理（不进行SMOTE）
X_test_processed = preprocessor.transform(X_test)

# ===================== 7. 构建随机森林模型 + 超参数调优 =====================
print("\n" + "=" * 60)
print("步骤4: 随机森林模型 + GridSearchCV超参数调优")
print("=" * 60)

# 定义随机森林分类器
rf = RandomForestClassifier(random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

# GridSearchCV（不使用多进程，n_jobs=1）
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

# 训练模型
grid_search.fit(X_train_smote, y_train_smote)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# 获取最佳模型
best_rf = grid_search.best_estimator_

# ===================== 8. 模型评估 =====================
print("\n" + "=" * 60)
print("步骤5: 模型评估（测试集）")
print("=" * 60)

# 预测
y_pred = best_rf.predict(X_test_processed)
y_prob = best_rf.predict_proba(X_test_processed)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n{'指标':<25} {'结果':<15}")
print("-" * 40)
print(f"{'准确率 (Accuracy)':<25} {accuracy:<15.4f}")
print(f"{'召回率 (Recall)':<25} {recall:<15.4f}")
print(f"{'精确率 (Precision)':<25} {precision:<15.4f}")
print(f"{'F1分数 (F1-score)':<25} {f1:<15.4f}")
print(f"{'AUC':<25} {auc:<15.4f}")
print("-" * 40)

# ===================== 9. 输出模型信息 =====================
print("\n" + "=" * 60)
print("步骤6: 模型信息汇总")
print("=" * 60)
print(f"最佳模型类型: {type(best_rf).__name__}")
print(f"特征数量: {X_train_smote.shape[1]}")
print(f"决策树数量 (n_estimators): {best_rf.n_estimators}")
print(f"最大深度 (max_depth): {best_rf.max_depth}")
print(f"最小分裂样本数 (min_samples_split): {best_rf.min_samples_split}")
print(f"最小叶节点样本数 (min_samples_leaf): {best_rf.min_samples_leaf}")
print(f"类别权重 (class_weight): {best_rf.class_weight}")

print("\n" + "=" * 60)
print("模型构建与评估完成！")
print("=" * 60)
