# -*- coding: utf-8 -*-

"""
随机森林分类模型
任务：
预测 TRUST 是否 >=16

环境：
PyCharm 2025.2.3
Python 3.x
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ==========================
# 1. 读取数据
# ==========================

file_path = "train_data.csv"

data = pd.read_csv(
    file_path,
    encoding="utf-8"
)

print("数据维度：", data.shape)
print(data.head())


# ==========================
# 2. 构建二分类目标变量
# ==========================

# TRUST为最后一列
target_column = "TRUST"

# TRUST >=16 定义为阳性类别
data["TRUST_binary"] = (data[target_column] >= 16).astype(int)


# 删除原始TRUST
data = data.drop(columns=[target_column])


# ==========================
# 3. 定义特征
# ==========================

features = [
    "SEX",
    "AGE",
    "DEPT",
    "DIAGNOSIS",
    "TPPA",
    "TP",
    "HIV",
    "WBC",
    "RBC",
    "PLT",
    "NC",
    "LY",
    "NLR"
]


X = data[features]
y = data["TRUST_binary"]


# ==========================
# 4. 划分训练集和测试集
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================
# 5. 定义变量类型
# ==========================

# One-Hot变量
onehot_features = [
    "SEX",
    "DEPT",
    "DIAGNOSIS"
]


# Ordinal变量
ordinal_feature = [
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


# ==========================
# 6. 数据预处理
# ==========================

# 连续变量：
# 中位数填充
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# One-Hot编码
onehot_transformer = Pipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# TPPA序数编码
ordinal_transformer = Pipeline(
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
            ordinal_feature
        )
    ]
)


# ==========================
# 7. 构建随机森林模型
# ==========================

rf_model = RandomForestClassifier(
    random_state=42
)


# ==========================
# 8. 构建包含SMOTE的Pipeline
# ==========================

pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
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
            rf_model
        )
    ]
)


# ==========================
# 9. 超参数调优
# ==========================

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
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=1,       # 不使用多进程
    verbose=1
)


# ==========================
# 10. 模型训练
# ==========================

print("\n开始模型训练...")

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


# ==========================
# 11. 测试集预测
# ==========================

best_model = grid_search.best_estimator_


y_pred = best_model.predict(
    X_test
)


# AUC需要概率
y_prob = best_model.predict_proba(
    X_test
)[:, 1]


# ==========================
# 12. 模型评价
# ==========================

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


# ==========================
# 13. 输出结果
# ==========================

print("\n========== 模型评价结果 ==========")

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

print(
    "================================="
)