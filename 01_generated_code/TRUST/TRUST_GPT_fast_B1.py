# -*- coding: utf-8 -*-

"""
随机森林模型预测 TRUST >=16
适用于 PyCharm 2025.2.3

功能：
1. 数据读取
2. 数据预处理
3. SMOTE处理类别不平衡
4. 随机森林建模
5. GridSearchCV调参
6. 模型性能评价
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
from imblearn.pipeline import Pipeline as ImbPipeline


# ===============================
# 1. 读取数据
# ===============================

file_path = "train_data.csv"

data = pd.read_csv(
    file_path,
    encoding="utf-8"
)

print("数据维度:")
print(data.shape)

print("\n数据前5行:")
print(data.head())


# ===============================
# 2. 构建二分类目标
# ===============================

# TRUST为最后一列
target = "TRUST"

# TRUST >=16定义为阳性
data["TRUST_binary"] = (
    data[target] >= 16
).astype(int)


# 删除原始TRUST
X = data.drop(
    columns=[target, "TRUST_binary"]
)

y = data["TRUST_binary"]


print("\n类别分布:")
print(y.value_counts())


# ===============================
# 3. 定义变量类型
# ===============================

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


# ===============================
# 4. 数据预处理
# ===============================

# 连续变量:
# 中位数填充
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# One-Hot编码
onehot_transformer = Pipeline(
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
ordinal_transformer = Pipeline(
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



# ===============================
# 5. 划分训练集和测试集
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n训练集:")
print(X_train.shape)

print("测试集:")
print(X_test.shape)



# ===============================
# 6. 建立随机森林模型
# ===============================

rf = RandomForestClassifier(
    random_state=42
)


# ===============================
# 7. 建立包含SMOTE的Pipeline
# ===============================

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
            rf
        )
    ]
)



# ===============================
# 8. 超参数搜索
# ===============================

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

    n_jobs=1,     # 不使用多进程

    verbose=1
)



# ===============================
# 9. 模型训练
# ===============================

print("\n开始模型训练...")

grid_search.fit(
    X_train,
    y_train
)


print("\n最佳参数:")
print(grid_search.best_params_)


print("\n最佳交叉验证AUC:")
print(grid_search.best_score_)



# ===============================
# 10. 测试集预测
# ===============================

best_model = grid_search.best_estimator_


y_pred = best_model.predict(
    X_test
)


# AUC需要概率
y_prob = best_model.predict_proba(
    X_test
)[:, 1]



# ===============================
# 11. 模型评价
# ===============================


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



print("\n分类报告:")
print(
    classification_report(
        y_test,
        y_pred
    )
)



# ===============================
# 12. 保存预测结果（可选）
# ===============================

result = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Prediction": y_pred,
        "Probability": y_prob
    }
)


result.to_csv(
    "TRUST_prediction_result.csv",
    index=False,
    encoding="utf-8"
)


print(
    "\n预测结果已保存: TRUST_prediction_result.csv"
)