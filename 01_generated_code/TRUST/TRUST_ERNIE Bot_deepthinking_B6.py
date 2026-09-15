# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 预测TRUST滴度是否≥16
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
                             f1_score, roc_auc_score, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# ========================== 1. 数据加载 ==========================
print("=" * 60)
print("步骤1: 加载数据")
print("=" * 60)

df = pd.read_csv('train_data.csv', encoding='utf-8')

print(f"数据集形状: {df.shape}")
print(f"数据集列名: {df.columns.tolist()}")
print(f"\n前5行数据预览:")
print(df.head())

# ========================== 2. 特征与目标分离 ==========================
print("\n" + "=" * 60)
print("步骤2: 特征与目标分离")
print("=" * 60)

# 目标列是最后一列 TRUST
X = df.iloc[:, :-1]  # 所有特征列
y = df.iloc[:, -1]   # 最后一列 TRUST

# 将目标转换为二分类：≥16 → 1，<16 → 0
y_binary = (y >= 16).astype(int)

print(f"目标列原始值分布:\n{y.value_counts().sort_index()}")
print(f"\n二分类目标分布:\n{y_binary.value_counts()}")
print(f"正样本(≥16)比例: {y_binary.mean():.2%}")

# ========================== 3. 定义特征列类型 ==========================
print("\n" + "=" * 60)
print("步骤3: 定义特征列类型")
print("=" * 60)

# 分类变量（独热编码）
categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']

# 分类变量（序数编码）
categorical_ordinal = ['TPPA']

# 连续变量
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

print(f"独热编码特征: {categorical_onehot}")
print(f"序数编码特征: {categorical_ordinal}")
print(f"连续变量特征: {continuous}")

# ========================== 4. 构建预处理管道 ==========================
print("\n" + "=" * 60)
print("步骤4: 构建预处理管道")
print("=" * 60)

# 连续变量预处理：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 分类变量预处理（独热编码）
categorical_onehot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 虽然说明无缺失，但为了管道完整性
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# 分类变量预处理（序数编码）
categorical_ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合所有预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal)
    ],
    remainder='drop'  # 丢弃未指定的列
)

print("预处理管道构建完成")

# ========================== 5. 划分训练集和测试集 ==========================
print("\n" + "=" * 60)
print("步骤5: 划分训练集和测试集")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"训练集正样本比例: {y_train.mean():.2%}")
print(f"测试集正样本比例: {y_test.mean():.2%}")

# ========================== 6. 构建完整管道（含SMOTE和随机森林） ==========================
print("\n" + "=" * 60)
print("步骤6: 构建完整管道（SMOTE + 随机森林）")
print("=" * 60)

# 定义随机森林分类器
rf_classifier = RandomForestClassifier(
    random_state=42,
    n_jobs=1  # 不使用多进程
)

# 定义超参数搜索空间
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 构建完整管道：预处理 → SMOTE → 随机森林
# 注意：SMOTE只能在数值特征上工作，所以需要先预处理再SMOTE
full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_classifier)
])

print("完整管道构建完成")
print(f"管道步骤: {full_pipeline.steps}")

# ========================== 7. 超参数调优（GridSearchCV） ==========================
print("\n" + "=" * 60)
print("步骤7: 超参数调优（GridSearchCV）")
print("=" * 60)

grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1,
    refit=True
)

print("开始网格搜索...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 获取最佳模型
best_model = grid_search.best_estimator_

# ========================== 8. 模型评估 ==========================
print("\n" + "=" * 60)
print("步骤8: 模型评估")
print("=" * 60)

# 在测试集上预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "-" * 40)
print("模型评估结果（测试集）")
print("-" * 40)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")

print("\n" + "-" * 40)
print("详细分类报告")
print("-" * 40)
print(classification_report(y_test, y_pred,
                            target_names=['<16', '≥16'],
                            digits=4))

# ========================== 9. 输出结果汇总 ==========================
print("\n" + "=" * 60)
print("步骤9: 结果汇总")
print("=" * 60)

results_summary = {
    'Accuracy': accuracy,
    'Recall': recall,
    'Precision': precision,
    'F1-score': f1,
    'AUC': auc
}

print("\n最终模型评估指标汇总:")
for metric, value in results_summary.items():
    print(f"  {metric:12s}: {value:.4f}")

print("\n" + "=" * 60)
print("模型构建与评估完成！")
print("=" * 60)
