# -*- coding: utf-8 -*-
"""
随机森林分类模型 - TRUST滴度预测（≥16 二分类）
适用环境：PyCharm 2025.2.3
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

# ========================== 1. 数据加载 ==========================
print("=" * 60)
print("步骤1：加载数据")
print("=" * 60)

df = pd.read_csv('train_data.csv', encoding='utf-8')
print(f"数据集形状：{df.shape}")
print(f"列名：{df.columns.tolist()}")

# ========================== 2. 特征与目标分离 ==========================
print("\n" + "=" * 60)
print("步骤2：分离特征与目标变量")
print("=" * 60)

# 目标列是最后一列 TRUST
feature_columns = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                  'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[feature_columns].copy()
y = df.iloc[:, -1].copy()  # 最后一列为TRUST

# 转换为二分类目标：TRUST >= 16 为1，否则为0
y_binary = (y >= 16).astype(int)
print(f"原始TRUST类别分布：\n{y.value_counts().sort_index()}")
print(f"二分类目标分布：\n{y_binary.value_counts()}")

# ========================== 3. 定义变量类型 ==========================
print("\n" + "=" * 60)
print("步骤3：定义变量类型")
print("=" * 60)

categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal = ['TPPA']                       # 序数编码
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

print(f"独热编码变量：{categorical_onehot}")
print(f"序数编码变量：{categorical_ordinal}")
print(f"连续变量：{continuous}")

# ========================== 4. 构建预处理流水线 ==========================
print("\n" + "=" * 60)
print("步骤4：构建预处理流水线")
print("=" * 60)

# 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量独热编码
categorical_onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# TPPA序数编码
categorical_ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

# 组合所有预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal)
    ]
)

print("预处理流水线构建完成")

# ========================== 5. 划分训练集和测试集 ==========================
print("\n" + "=" * 60)
print("步骤5：划分训练集和测试集")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"训练集大小：{X_train.shape[0]}")
print(f"测试集大小：{X_test.shape[0]}")
print(f"训练集目标分布：\n{pd.Series(y_train).value_counts()}")
print(f"测试集目标分布：\n{pd.Series(y_test).value_counts()}")

# ========================== 6. 应用预处理 + SMOTE ==========================
print("\n" + "=" * 60)
print("步骤6：数据预处理与SMOTE过采样")
print("=" * 60)

# 先对训练集进行预处理
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"预处理后训练集特征维度：{X_train_processed.shape}")
print(f"预处理后测试集特征维度：{X_test_processed.shape}")

# 使用SMOTE处理类别不平衡
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print(f"SMOTE后训练集大小：{X_train_smote.shape[0]}")
print(f"SMOTE后目标分布：\n{pd.Series(y_train_smote).value_counts()}")

# ========================== 7. 构建随机森林模型并超参数调优 ==========================
print("\n" + "=" * 60)
print("步骤7：随机森林模型构建与超参数调优")
print("=" * 60)

# 定义随机森林分类器
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=1  # 不使用多进程
)

# 定义超参数搜索空间
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

# 网格搜索（不使用多进程）
grid_search = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

print("开始网格搜索...")
grid_search.fit(X_train_smote, y_train_smote)

print(f"\n最佳参数：{grid_search.best_params_}")
print(f"最佳交叉验证AUC得分：{grid_search.best_score_:.4f}")

# 使用最佳模型
best_model = grid_search.best_estimator_

# ========================== 8. 模型评估 ==========================
print("\n" + "=" * 60)
print("步骤8：模型评估")
print("=" * 60)

# 在测试集上预测
y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("模型评估结果（测试集）")
print("=" * 60)
print(f"准确率（Accuracy）：  {accuracy:.4f}")
print(f"召回率（Recall）：    {recall:.4f}")
print(f"精确率（Precision）： {precision:.4f}")
print(f"F1分数（F1-score）：  {f1:.4f}")
print(f"AUC：                 {auc:.4f}")
print("=" * 60)

# ========================== 9. 输出总结 ==========================
print("\n" + "=" * 60)
print("模型构建完成！")
print("=" * 60)
print(f"最佳模型参数：")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\n最终评估指标：")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  F1-score:  {f1:.4f}")
print(f"  AUC:       {auc:.4f}")
print("=" * 60)
