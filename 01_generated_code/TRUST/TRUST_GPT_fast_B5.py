# -*- coding: utf-8 -*-

"""
随机森林模型预测 TRUST >=16

环境：
PyCharm 2025.2.3
Python 3.x

功能：
1. 数据读取
2. 数据预处理
3. SMOTE类别平衡
4. 随机森林建模
5. GridSearchCV参数优化
6. 模型评价
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
from imblearn.pipeline import Pipeline as ImbPipeline



# ==============================
# 1. 读取数据
# ==============================

file_path = "train_data.csv"

data = pd.read_csv(
    file_path,
    encoding="utf-8"
)

print("数据维度：", data.shape)
print("\n数据前5行：")
print(data.head())



# ==============================
# 2. 构建二分类目标
# TRUST >=16 定义为1，否则0
# ==============================

data["TARGET"] = np.where(
    data["TRUST"] >= 16,
    1,
    0
)


# 删除原始TRUST
data = data.drop(
    columns=["TRUST"]
)


# ==============================
# 3. 定义特征
# ==============================


# 分类变量
onehot_features = [
    "SEX",
    "DEPT",
    "DIAGNOSIS"
]


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


X = data.drop(
    columns=["TARGET"]
)

y = data["TARGET"]


print("\n类别比例：")
print(y.value_counts())



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
# 5. 数据预处理
# ==============================


# 连续变量：
# 中位数填充
numeric_transformer = Pipeline(
    steps=[
        (
            "median_imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# SEX DEPT DIAGNOSIS
# One-Hot编码

onehot_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# TPPA
# Ordinal编码

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
# 6. 建立模型Pipeline
# ==============================

rf = RandomForestClassifier(
    random_state=42
)


model_pipeline = ImbPipeline(
    steps=[

        # 数据预处理
        (
            "preprocessor",
            preprocessor
        ),

        # SMOTE
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



# ==============================
# 7. 超参数搜索
# ==============================


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

    n_jobs=1,       # 不使用多进程

    verbose=1

)



# ==============================
# 8. 模型训练
# ==============================

print("\n开始模型训练...")

grid_search.fit(
    X_train,
    y_train
)


print("\n最佳参数：")
print(grid_search.best_params_)


print("\n最佳交叉验证AUC:")
print(grid_search.best_score_)



# ==============================
# 9. 测试集预测
# ==============================


best_model = grid_search.best_estimator_


y_pred = best_model.predict(
    X_test
)


# 获取概率
y_prob = best_model.predict_proba(
    X_test
)[:,1]



# ==============================
# 10. 模型评价
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



# ==============================
# 11. 保存预测结果
# ==============================


result = pd.DataFrame(
    {
        "True_Label": y_test.values,
        "Predicted_Label": y_pred,
        "Predicted_Probability": y_prob
    }
)


result.to_csv(
    "RF_prediction_results.csv",
    index=False,
    encoding="utf-8"
)


print(
    "\n预测结果已保存：RF_prediction_results.csv"
)