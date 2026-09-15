# -*- coding: utf-8 -*-
"""
随机森林分类模型构建与评估
数据集: CrAg_train.csv
目标: 预测CSF-T是否 >= 20 (二分类)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings('ignore')

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("1. 加载数据...")
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# ==================== 2. 特征与目标分离 ====================
print("\n" + "=" * 60)
print("2. 特征与目标分离...")

# 特征列列表
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']

# 目标列是最后一列 'CSF-T'
target_col = 'CSF-T'

# 提取特征和目标
X = df[feature_cols].copy()
y = df[target_col].copy()

# 将目标转换为二分类: 是否 >= 20
y_binary = (y >= 20).astype(int)
print(f"目标列原始值分布:\n{y.value_counts().sort_index()}")
print(f"\n二分类目标分布:\n{y_binary.value_counts()}")
print(f"正类(>=20)比例: {y_binary.mean():.4f}")

# ==================== 3. 划分训练集和验证集 (4:1) ====================
print("\n" + "=" * 60)
print("3. 划分训练集和验证集 (4:1)...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"训练集形状: {X_train.shape}, 正类比例: {y_train.mean():.4f}")
print(f"验证集形状: {X_val.shape}, 正类比例: {y_val.mean():.4f}")

# ==================== 4. 定义列类型 ====================
print("\n" + "=" * 60)
print("4. 定义列类型...")

# 分类变量
categorical_onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal_cols = ['SER-T', 'Ink staining']

# 连续变量
numeric_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

print(f"独热编码变量: {categorical_onehot_cols}")
print(f"序数编码变量: {categorical_ordinal_cols}")
print(f"连续变量: {numeric_cols}")

# ==================== 5. 构建预处理Pipeline ====================
print("\n" + "=" * 60)
print("5. 构建预处理Pipeline...")

# 5.1 数值特征预处理: 中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 5.2 独热编码: 无缺失值
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 5.3 序数编码: 无缺失值
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 5.4 组合预处理器
# 注意: 顺序为 数值 → 独热 → 序数 (按ColumnTransformer的内部顺序执行)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('onehot', onehot_transformer, categorical_onehot_cols),
        ('ordinal', ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ==================== 6. 构建完整Pipeline (含SMOTE) ====================
print("\n" + "=" * 60)
print("6. 构建完整Pipeline (SMOTE嵌入交叉验证内部)...")

# 使用 imblearn 的 Pipeline，确保 SMOTE 只应用于训练折
full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ==================== 7. 超参数调优 ====================
print("\n" + "=" * 60)
print("7. 超参数调优 (GridSearchCV)...")

# 定义参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 使用分层K折交叉验证，将SMOTE嵌入内部
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

print("开始GridSearchCV训练...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ==================== 8. 模型评估 ====================
print("\n" + "=" * 60)
print("8. 模型评估...")

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

print("\n验证集评估结果:")
print("-" * 40)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC (ROC曲线下面积): {auc:.4f}")
print("-" * 40)

# ==================== 9. 输出混淆矩阵相关统计 ====================
print("\n" + "=" * 60)
print("9. 详细分类结果...")

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_val, y_pred)
print(f"混淆矩阵:\n{cm}")
print(f"\n分类报告:")
print(classification_report(y_val, y_pred, target_names=['<20', '>=20']))

# ==================== 10. 特征重要性 (可选) ====================
print("\n" + "=" * 60)
print("10. 特征重要性 (Top 10)...")

# 获取最佳模型中的分类器
classifier = best_model.named_steps['classifier']

# 获取特征名称
preprocessor_fitted = best_model.named_steps['preprocessor']
feature_names = []

# 数值特征
feature_names.extend(numeric_cols)

# 独热编码特征
onehot_encoder = preprocessor_fitted.named_transformers_['onehot']
if hasattr(onehot_encoder, 'get_feature_names_out'):
    onehot_names = onehot_encoder.get_feature_names_out(categorical_onehot_cols)
    feature_names.extend(onehot_names)
else:
    # 兼容旧版本
    for col in categorical_onehot_cols:
        categories = onehot_encoder.categories_
        for cat in categories:
            feature_names.append(f"{col}_{cat}")

# 序数编码特征
feature_names.extend(categorical_ordinal_cols)

# 获取特征重要性
importances = classifier.feature_importances_

# 创建特征重要性DataFrame
feature_importance_df = pd.DataFrame({
    'feature': feature_names[:len(importances)],
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 重要特征:")
print(feature_importance_df.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("模型训练与评估完成!")