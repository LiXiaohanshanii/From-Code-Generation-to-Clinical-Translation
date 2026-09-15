# -*- coding: utf-8 -*-
"""
随机森林分类模型构建脚本
任务：预测TRUST滴度是否 >= 16（二分类）
环境：PyCharm 2025.2.3 / Python 3.x
作者：检验科数据分析组
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
)

# ===================== 1. 数据加载 =====================
print("=" * 60)
print("正在加载数据集...")
df = pd.read_csv("train_data.csv", encoding="utf-8")
print(f"数据集形状: {df.shape}")
print(f"目标列(TRUST)原始分布:\n{df['TRUST'].value_counts()}")

# ===================== 2. 标签构建（二分类转换）=====================
# TRUST为滴度值，将 >=16 定义为阳性(1)，<16 定义为阴性(0)
y = (df["TRUST"] >= 16).astype(int)
print(f"\n二分类标签分布:\n{y.value_counts()}")

# ===================== 3. 特征工程定义 =====================
# 分类变量
cat_onehot_cols = ["SEX", "DEPT", "DIAGNOSIS"]   # 独热编码
cat_ordinal_cols = ["TPPA"]                        # 序数编码

# 连续变量
num_cols = ["AGE", "TP", "HIV", "WBC", "RBC", "PLT", "NC", "LY", "NLR"]

# 选择所有特征列（排除目标列TRUST）
feature_cols = cat_onehot_cols + cat_ordinal_cols + num_cols
X = df[feature_cols].copy()

# ===================== 4. 划分训练集与测试集 =====================
# 先划分再做过采样，防止SMOTE引入数据泄露
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# ===================== 5. 构建预处理+模型Pipeline =====================
# 5.1 预处理器：分别对不同列类型做对应处理
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_onehot_cols),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_ordinal_cols),
        ("num_impute", SimpleImputer(strategy="median"), num_cols),
    ],
    remainder="drop"  # 丢弃未指定的列
)

# 5.2 使用imblearn的Pipeline以支持SMOTE步骤
# 注意：SMOTE放在预处理之后、模型之前，确保只对数值化后的特征做过采样
model_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])

# ===================== 6. 超参数调优（GridSearchCV，单进程）=====================
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1],
    "classifier__class_weight": ["balanced"],
}

print("\n正在进行网格搜索超参数调优（单进程模式）...")
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="f1",           # 不平衡数据以F1为主要优化指标
    cv=5,
    n_jobs=1,               # 明确指定不使用多进程
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n最佳参数组合: {grid_search.best_params_}")
print(f"交叉验证最佳F1分数: {grid_search.best_score_:.4f}")

# ===================== 7. 模型评估 =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("测试集模型评估结果")
print("=" * 60)
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC曲线下面积:      {auc:.4f}")
print("=" * 60)
