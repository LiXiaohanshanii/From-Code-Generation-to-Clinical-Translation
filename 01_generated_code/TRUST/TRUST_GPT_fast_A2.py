# -*- coding: utf-8 -*-

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

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ==============================
# 1. 读取数据
# ==============================

file_path = "train_data.csv"

data = pd.read_csv(
    file_path,
    encoding="utf-8"
)

print("数据维度：", data.shape)
print(data.head())


# ==============================
# 2. 构建二分类目标变量
# ==============================

# TRUST >=16 定义为阳性类别
data["TRUST_binary"] = (data["TRUST"] >= 16).astype(int)


# ==============================
# 3. 定义特征和目标
# ==============================

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


# ==============================
# 4. 划分训练集和测试集
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# 5. 定义变量类型
# ==============================

# One-Hot编码变量
onehot_features = [
    "SEX",
    "DEPT",
    "DIAGNOSIS"
]


# Ordinal编码变量
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


# ==============================
# 6. 数据预处理
# ==============================

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


# One-Hot Encoder
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


# Ordinal Encoder
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


# ==============================
# 7. 建立随机森林模型
# ==============================

rf = RandomForestClassifier(
    random_state=42
)


# ==============================
# 8. 建立Pipeline
# 包含：
# 数据处理
# SMOTE
# 随机森林
# ==============================

model_pipeline = Pipeline(
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
            rf
        )
    ]
)


# ==============================
# 9. 超参数搜索
# 不使用多进程
# ==============================

param_grid = {

    "classifier__n_estimators": [
        100,
        200
    ],

    "classifier__max_depth": [
        10
    ],

    "classifier__min_samples_split": [
        2
    ],

    "classifier__min_samples_leaf": [
        1
    ],

    "classifier__class_weight": [
        "balanced"
    ]
}


grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=1,        # 不使用多进程
    verbose=1
)


# ==============================
# 10. 模型训练
# ==============================

print("\n开始训练模型...")

grid_search.fit(
    X_train,
    y_train
)


print("\n最佳参数：")
print(grid_search.best_params_)


print("\n最佳交叉验证AUC：")
print(grid_search.best_score_)


# ==============================
# 11. 测试集预测
# ==============================

best_model = grid_search.best_estimator_


y_pred = best_model.predict(
    X_test
)


# 预测概率（用于AUC）
y_prob = best_model.predict_proba(
    X_test
)[:, 1]


# ==============================
# 12. 模型评价
# ==============================

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
print("模型评价结果")
print("==============================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")


print("\n分类报告：")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)