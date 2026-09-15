# -*- coding: utf-8 -*-
"""
随机森林分类模型构建与评估
适用环境：PyCharm 2025.2.3
任务：预测TRUST是否≥16（二分类）
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

# ==================== 1. 数据加载 ====================
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征和目标
X = data.iloc[:, :-1]  # 所有特征列
y = data.iloc[:, -1]   # 目标列 TRUST

# ==================== 2. 目标变量转换（二分类：≥16 vs <16） ====================
y = (y >= 16).astype(int)

# ==================== 3. 定义特征列类型 ====================
# 分类变量（独热编码）
categorical_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']
# 分类变量（序数编码）
categorical_ordinal = ['TPPA']
# 连续变量
continuous = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ==================== 4. 构建预处理管道 ====================
# 连续变量：中位数填充
continuous_transformer = SimpleImputer(strategy='median')

# 分类变量（独热编码）
categorical_ohe_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# 分类变量（序数编码）
categorical_ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 组合所有预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous),
        ('cat_ohe', categorical_ohe_transformer, categorical_ohe),
        ('cat_ord', categorical_ordinal_transformer, categorical_ordinal)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ==================== 5. 数据集划分 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================== 6. 预处理训练数据 ====================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==================== 7. SMOTE处理分类不平衡 ====================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print(f"SMOTE前训练集类别分布: {np.bincount(y_train)}")
print(f"SMOTE后训练集类别分布: {np.bincount(y_train_smote)}")

# ==================== 8. 构建随机森林模型与超参数调优 ====================
rf_classifier = RandomForestClassifier(random_state=42)

# 超参数网格
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

# 构建完整管道（预处理 + 模型）
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_classifier)
])

# GridSearchCV超参数调优（不使用多进程）
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

# 拟合模型
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("\n最佳超参数:")
print(grid_search.best_params_)
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# ==================== 9. 模型评估 ====================
# 使用最佳模型进行预测
best_model = grid_search.best_estimator_

# 对测试集进行预测（获取概率用于AUC计算）
# 注意：pipeline内部会自动处理预处理和SMOTE（仅训练时），测试时只做预处理
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*50)
print("模型评估结果（测试集）")
print("="*50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("="*50)
