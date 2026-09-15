# -*- coding: utf-8 -*-
"""
随机森林分类模型构建与评估
适用环境：PyCharm 2025.2.3
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

# ============================================================
# 1. 数据加载
# ============================================================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征和目标
X = data.iloc[:, :-1]  # 所有特征列
y = data.iloc[:, -1]   # 最后一列为TRUST

# ============================================================
# 2. 目标变量转换为二分类（TRUST >= 16）
# ============================================================
y_binary = (y >= 16).astype(int)

# ============================================================
# 3. 定义特征列类型
# ============================================================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
categorical_ordinal_cols = ['TPPA']                     # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# ============================================================
# 4. 构建预处理管道
# ============================================================
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量（独热编码）
categorical_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 分类变量（序数编码）
categorical_ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_cols),
        ('cat_onehot', categorical_onehot_transformer, categorical_onehot_cols),
        ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'
)

# ============================================================
# 5. 构建完整管道（含SMOTE和随机森林）
# ============================================================
# 随机森林分类器
rf_classifier = RandomForestClassifier(random_state=42)

# 使用 ImbPipeline 以确保 SMOTE 在正确阶段执行
# 注意：SMOTE 需要在划分数据之后应用，因此我们先划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 对训练集进行预处理
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# SMOTE 处理分类不平衡（仅对训练集）
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

# ============================================================
# 6. 超参数调优（GridSearchCV，不使用多进程）
# ============================================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    verbose=1
)

grid_search.fit(X_train_smote, y_train_smote)

# 最佳模型
best_model = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ============================================================
# 7. 模型评估
# ============================================================
y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*50)
print("模型评估结果：")
print("="*50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("="*50)
