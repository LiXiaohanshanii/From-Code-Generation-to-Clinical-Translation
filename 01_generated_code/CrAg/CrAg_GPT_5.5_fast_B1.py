import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

# ==========================
# 1. 读取数据
# ==========================

data = pd.read_csv("CrAg_train.csv", encoding="utf-8")

# ==========================
# 2. 构建二分类标签
# CSF-T >=20 为1，否则为0
# ==========================

data["CSF-T"] = pd.to_numeric(data["CSF-T"], errors="coerce")
data["Target"] = (data["CSF-T"] >= 20).astype(int)

# 删除原始目标列
data = data.drop(columns=["CSF-T"])

# ==========================
# 3. 特征
# ==========================

X = data.drop(columns=["Target"])
y = data["Target"]

# 连续变量
numeric_features = [
    "CL",
    "GLU",
    "Protein",
    "RBC",
    "AGE"
]

# One-Hot变量
onehot_features = [
    "Color",
    "Transparency",
    "SEX",
    "DEPT",
    "DIAGNOSIS"
]

# Ordinal变量
ordinal_features = [
    "SER-T",
    "Ink staining"
]

# ==========================
# 4. 预处理
# ==========================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

onehot_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

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
        ("num", numeric_transformer, numeric_features),
        ("onehot", onehot_transformer, onehot_features),
        ("ordinal", ordinal_transformer, ordinal_features)
    ]
)

# ==========================
# 5. 划分训练集测试集
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==========================
# 6. 建立Pipeline
# 注意：SMOTE只作用于训练集
# ==========================

pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42))
    ]
)

# ==========================
# 7. 超参数搜索
# ==========================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1],
    "classifier__class_weight": ["balanced"]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=1,          # 不使用多进程
    verbose=1
)

# ==========================
# 8. 训练模型
# ==========================

grid_search.fit(X_train, y_train)

print("=" * 60)
print("Best Parameters:")
print(grid_search.best_params_)
print()

print("Best CV F1:")
print(grid_search.best_score_)
print("=" * 60)

# ==========================
# 9. 测试集预测
# ==========================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]

# ==========================
# 10. 模型评估
# ==========================

accuracy = accuracy_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_prob)

print("\nModel Performance")
print("-" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")