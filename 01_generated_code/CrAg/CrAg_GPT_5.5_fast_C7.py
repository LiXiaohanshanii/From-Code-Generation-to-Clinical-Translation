import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline
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
# 读取数据
# ==========================
data = pd.read_csv("CrAg_train.csv", encoding="utf-8")

# ==========================
# 构建二分类标签
# CSF-T >=20 为1，否则0
# ==========================
data["Target"] = (data["CSF-T"] >= 20).astype(int)

# ==========================
# 特征列
# ==========================
feature_cols = [
    "SER-T",
    "CL",
    "GLU",
    "Ink staining",
    "Color",
    "Protein",
    "Transparency",
    "RBC",
    "SEX",
    "AGE",
    "DEPT",
    "DIAGNOSIS"
]

X = data[feature_cols]
y = data["Target"]

# ==========================
# 分类变量
# ==========================
numeric_features = [
    "CL",
    "GLU",
    "Protein",
    "RBC",
    "AGE"
]

onehot_features = [
    "Color",
    "Transparency",
    "SEX",
    "DEPT",
    "DIAGNOSIS"
]

ordinal_features = [
    "SER-T",
    "Ink staining"
]

# ==========================
# 数值变量
# 中位数填充
# ==========================
numeric_transformer = SkPipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# ==========================
# One-Hot编码
# 分类变量无缺失
# ==========================
onehot_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# ==========================
# Ordinal编码
# ==========================
ordinal_transformer = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

# ==========================
# ColumnTransformer
# 顺序：
# 数值 -> OneHot -> Ordinal
# ==========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("onehot", onehot_transformer, onehot_features),
        ("ordinal", ordinal_transformer, ordinal_features)
    ]
)

# ==========================
# 划分训练集和验证集
# ==========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# 建立Pipeline
# 注意：
# SMOTE放在Pipeline内部
# 防止数据泄露
# ==========================
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42))
    ]
)

# ==========================
# 参数网格
# ==========================
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1],
    "classifier__class_weight": ["balanced"]
}

# ==========================
# 五折交叉验证
# ==========================
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ==========================
# GridSearchCV
# 单线程
# ==========================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=1,
    refit=True
)

# ==========================
# 模型训练
# ==========================
grid_search.fit(X_train, y_train)

# ==========================
# 最佳模型
# ==========================
best_model = grid_search.best_estimator_

print("=" * 60)
print("Best Parameters:")
print(grid_search.best_params_)
print("Best CV F1:")
print(grid_search.best_score_)
print("=" * 60)

# ==========================
# 验证集预测
# ==========================
y_pred = best_model.predict(X_valid)

y_prob = best_model.predict_proba(X_valid)[:, 1]

# ==========================
# 评价指标
# ==========================
accuracy = accuracy_score(y_valid, y_pred)

recall = recall_score(y_valid, y_pred)

precision = precision_score(y_valid, y_pred)

f1 = f1_score(y_valid, y_pred)

auc = roc_auc_score(y_valid, y_prob)

print("\nValidation Performance")
print("-" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")