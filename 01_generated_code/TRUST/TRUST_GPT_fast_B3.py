# -*- coding: utf-8 -*-

"""
随机森林分类模型
任务：
预测 TRUST >=16

环境：
PyCharm 2025.2.3
Python 3.x
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# =========================
# 1. 读取数据
# =========================

file_path = "train_data.csv"

df = pd.read_csv(
    file_path,
    encoding="utf-8"
)

print("数据维度：", df.shape)
print(df.head())


# =========================
# 2. 构建目标变量
# =========================

# TRUST >=16 定义为阳性
df["TARGET"] = (df["TRUST"] >= 16).astype(int)


# 删除原始TRUST
X = df.drop(
    columns=["TRUST", "TARGET"]
)

y = df["TARGET"]


# =========================
# 3. 定义变量类型
# =========================

# One-Hot变量
onehot_features = [
    "SEX",
    "DEPT",
    "DIAGNOSIS"
]


# Ordinal变量
ordinal_features = [
    "TPPA"
]


# 连续变量
numeric_features = [
    "AGE",
    "TP",
    "HIV",
    "WBC",
    "RBC",
    "PLT",
    "NC",
    "LY",
    "NLR"
]


# =========================
# 4. 数据预处理
# =========================

# 连续变量：
# 中位数填充
numeric_transformer = SklearnPipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# One-Hot编码
onehot_transformer = SklearnPipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# TPPA序数编码
ordinal_transformer = SklearnPipeline(
    steps=[
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ]
)


# 合并预处理
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),

        (
            "onehot",
            onehot_transformer,
            onehot_features
        ),

        (
            "ordinal",
            ordinal_transformer,
            ordinal_features
        )
    ]
)



# =========================
# 5. 划分训练集和测试集
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n训练集类别分布：")
print(y_train.value_counts())

print("\n测试集类别分布：")
print(y_test.value_counts())



# =========================
# 6. 构建随机森林模型
# =========================

rf = RandomForestClassifier(
    random_state=42
)


# Pipeline:
# 预处理 → SMOTE → RF

model_pipeline = Pipeline(
    steps=[
        (
            "preprocess",
            preprocessor
        ),

        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        (
            "classifier",
            rf
        )
    ]
)



# =========================
# 7. 超参数搜索
# =========================

param_grid = {

    "classifier__n_estimators":
        [100, 200],

    "classifier__max_depth":
        [10],

    "classifier__min_samples_split":
        [2],

    "classifier__min_samples_leaf":
        [1],

    "classifier__class_weight":
        ["balanced"]
}


grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,

    # 五折交叉验证
    cv=5,

    scoring="roc_auc",

    # 不使用多进程
    n_jobs=1,

    verbose=1
)



# =========================
# 8. 模型训练
# =========================

print("\n开始训练模型...")

grid_search.fit(
    X_train,
    y_train
)


print("\n最佳参数：")
print(grid_search.best_params_)

print(
    "\n最佳交叉验证AUC:",
    grid_search.best_score_
)



# =========================
# 9. 测试集预测
# =========================

best_model = grid_search.best_estimator_


y_pred = best_model.predict(
    X_test
)


# 获取概率
y_prob = best_model.predict_proba(
    X_test
)[:, 1]



# =========================
# 10. 模型评价
# =========================

accuracy = accuracy_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)


auc = roc_auc_score(
    y_test,
    y_prob
)



print("\n========== 模型评价 ==========")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

print(
    f"AUC       : {auc:.4f}"
)



print("\n分类报告：")

print(
    classification_report(
        y_test,
        y_pred
    )
)



# =========================
# 11. 保存模型（可选）
# =========================

import joblib


joblib.dump(
    best_model,
    "random_forest_TRUST_model.pkl"
)


print(
    "\n模型已保存：random_forest_TRUST_model.pkl"
)