# -*- coding: utf-8 -*-
"""
随机森林分类模型构建与评估
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ======================== 1. 数据加载 ========================
print("=" * 50)
print("步骤1: 加载数据")
print("=" * 50)

df = pd.read_csv('train_data.csv', encoding='utf-8')

print(f"数据集形状: {df.shape}")
print(f"特征列: {df.columns.tolist()}")

# ======================== 2. 定义特征与目标 ========================
feature_columns = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                   'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
target_column = 'TRUST'  # 最后一列

X = df[feature_columns]
y = df[target_column]

# 二分类任务: TRUST >= 16 为正类(1), < 16 为负类(0)
y_binary = (y >= 16).astype(int)
print(f"\n正类(>=16)样本数: {(y_binary == 1).sum()}")
print(f"负类(<16)样本数: {(y_binary == 0).sum()}")

# ======================== 3. 定义变量类型 ========================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal_cols = ['TPPA']  # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ======================== 4. 构建预处理管道 ========================
print("\n" + "=" * 50)
print("步骤2: 构建预处理管道")
print("=" * 50)

# 连续变量缺失值填充（中位数）
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量独热编码
onehot_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# TPPA序数编码
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 组合预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('onehot', onehot_transformer, categorical_onehot_cols),
        ('ordinal', ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ======================== 5. 构建完整管道（含SMOTE + 随机森林） ========================
print("\n" + "=" * 50)
print("步骤3: 构建含SMOTE和随机森林的完整管道")
print("=" * 50)

# 随机森林分类器
rf_classifier = RandomForestClassifier(random_state=42)

# 完整管道：预处理 -> SMOTE -> 随机森林
full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_classifier)
])

# ======================== 6. 划分训练集和测试集 ========================
print("\n" + "=" * 50)
print("步骤4: 划分训练集和测试集")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# ======================== 7. 超参数调优 ========================
print("\n" + "=" * 50)
print("步骤5: 网格搜索超参数调优")
print("=" * 50)

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,  # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# ======================== 8. 模型评估 ========================
print("\n" + "=" * 50)
print("步骤6: 在测试集上评估模型")
print("=" * 50)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "-" * 40)
print("模型评估结果:")
print("-" * 40)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("-" * 40)

# ======================== 9. 输出详细结果 ========================
print("\n" + "=" * 50)
print("步骤7: 输出预测结果对比（前20条）")
print("=" * 50)

result_df = pd.DataFrame({
    '真实值': y_test.values,
    '预测值': y_pred,
    '预测概率': y_prob
})
print(result_df.head(20).to_string(index=False))

print("\n程序运行完毕。")
