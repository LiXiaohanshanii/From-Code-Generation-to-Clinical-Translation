# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 预测TRUST是否≥16
适用环境：PyCharm 2025.2.3
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("步骤1：加载数据")
print("=" * 60)

df = pd.read_csv('train_data.csv', encoding='utf-8')

print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 分离特征和目标变量
X = df.iloc[:, :-1]  # 所有特征列（最后一列之前）
y = df.iloc[:, -1]   # 目标列TRUST

print(f"\n特征列数: {X.shape[1]}")
print(f"目标变量分布:\n{y.value_counts().sort_index()}")

# ==================== 2. 目标变量转换（二分类：≥16 vs <16）====================
print("\n" + "=" * 60)
print("步骤2：目标变量转换为二分类")
print("=" * 60)

y_binary = (y >= 16).astype(int)
print(f"二分类目标分布:\n{y_binary.value_counts()}")

# ==================== 3. 定义特征列类型 ====================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal_cols = ['TPPA']                      # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

print(f"\n独热编码列: {categorical_onehot_cols}")
print(f"序数编码列: {categorical_ordinal_cols}")
print(f"连续变量列: {continuous_cols}")

# ==================== 4. 构建预处理管道 ====================
print("\n" + "=" * 60)
print("步骤3：构建预处理管道")
print("=" * 60)

# 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量独热编码
categorical_onehot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 虽然无缺失，但为了管道完整性
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 分类变量序数编码
categorical_ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])

# 组合所有预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot_cols),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ==================== 5. 构建完整Pipeline（预处理 + SMOTE + 模型）====================
print("\n" + "=" * 60)
print("步骤4：构建完整模型管道")
print("=" * 60)

# SMOTE + 随机森林
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ==================== 6. 超参数调优 ====================
print("\n" + "=" * 60)
print("步骤5：超参数调优（GridSearchCV）")
print("=" * 60)

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,  # 不使用多进程
    verbose=1,
    refit=True
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 执行网格搜索
print("\n开始网格搜索...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# ==================== 7. 模型评估 ====================
print("\n" + "=" * 60)
print("步骤6：模型评估")
print("=" * 60)

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n模型评估结果:")
print(f"  准确率 (Accuracy):  {accuracy:.4f}")
print(f"  召回率 (Recall):    {recall:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  F1分数 (F1-score):  {f1:.4f}")
print(f"  AUC:                {auc:.4f}")

# ==================== 8. 输出结果汇总 ====================
print("\n" + "=" * 60)
print("模型构建与评估完成！")
print("=" * 60)
