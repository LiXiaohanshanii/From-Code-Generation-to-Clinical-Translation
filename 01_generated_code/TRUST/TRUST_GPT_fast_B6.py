# -*- coding: utf-8 -*-

"""
随机森林模型预测 TRUST >=16

适用于 PyCharm 2025.2.3
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

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


# ============================
# 1. 数据读取
# ============================

file_path = "train_data.csv"

data = pd.read_csv(
    file_path,
    encoding="utf-8"
)

print("数据维度：", data.shape)
print(data.head())


# ============================
# 2. 构建二分类目标
# ============================

# TRUST >=16 定义为阳性
data["TARGET"] = (data["TRUST"] >= 16).astype(int)


# 删除原始TRUST
X = data.drop(
    columns=["TRUST", "TARGET"]
)

y = data["TARGET"]


print("\n类别分布：")
print(y.value_counts())


# ============================
# 3. 定义变量类型
# ============================


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
continuous_features = [
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


# ============================
# 4. 数据预处理
# ============================


# 连续变量：
# 中位数填补
continuous_transformer = SklearnPipeline(
    steps=[
        (
            "median_imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# One-Hot编码
onehot_transformer = SklearnPipeline(
    steps=[
        (
            "onehot",
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
            "ordinal",
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
            "continuous",
            continuous_transformer,
            continuous_features
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


# ============================
# 5. 划分训练集和测试集
# ============================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================
# 6. 建立随机森林模型
# ============================


rf = RandomForestClassifier(
    random_state=42
)


# ============================
# 7. 建立完整Pipeline
# ============================


model_pipeline = Pipeline(
    steps=[

        # 数据预处理
        (
            "preprocessor",
            preprocessor
        ),

        # SMOTE处理不平衡
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),

        # 随机森林
        (
            "classifier",
            rf
        )
    ]
)


# ============================
# 8. 参数优化
# ============================


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

    scoring="roc_auc",

    cv=5,

    n_jobs=1,

    verbose=2

)


# ============================
# 9. 模型训练
# ============================


print("\n开始模型训练...")

grid_search.fit(
    X_train,
    y_train
)


print("\n最佳参数：")
print(grid_search.best_params_)


print("\n最佳交叉验证AUC:")
print(grid_search.best_score_)



# ============================
# 10. 模型预测
# ============================


best_model = grid_search.best_estimator_


y_pred = best_model.predict(
    X_test
)


# AUC需要概率
y_prob = best_model.predict_proba(
    X_test
)[:, 1]


# ============================
# 11. 模型评价
# ============================


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


print("\n==============================")
print("随机森林模型评价结果")
print("==============================")

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