# -*- coding: utf-8 -*-
"""
随机森林分类模型 - 梅毒TRUST滴度预测（≥16 vs <16）
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
import warnings

warnings.filterwarnings('ignore')

# ========================== 1. 数据加载 ==========================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 分离特征与目标
X = df.iloc[:, :-1]          # 特征列（除最后一列）
y = df.iloc[:, -1]           # 目标列 TRUST（最后一列）

# 构建二分类目标：TRUST >= 16 为1，否则为0
y_binary = (y >= 16).astype(int)

# ========================== 2. 定义列类型 ==========================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']   # 独热编码
categorical_ordinal_cols = ['TPPA']                       # 序数编码
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

# ========================== 3. 构建预处理管道 ==========================
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
         categorical_onehot_cols),
        ('ordinal', OrdinalEncoder(),
         categorical_ordinal_cols),
        ('continuous', SimpleImputer(strategy='median'),
         continuous_cols)
    ],
    remainder='drop'
)

# ========================== 4. 划分训练集与测试集 ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ========================== 5. 对训练集进行预处理 + SMOTE ==========================
X_train_processed = preprocessor.fit_transform(X_train)

# SMOTE 处理分类不平衡
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

print(f"SMOTE前训练集类别分布: {np.bincount(y_train)}")
print(f"SMOTE后训练集类别分布: {np.bincount(y_train_smote)}")

# ========================== 6. 构建模型并超参数调优 ==========================
rf_clf = RandomForestClassifier(random_state=42, n_jobs=1)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=0
)

grid_search.fit(X_train_smote, y_train_smote)

# 输出最佳参数
print(f"\n最佳超参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ========================== 7. 在测试集上评估 ==========================
X_test_processed = preprocessor.transform(X_test)

y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

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
