import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# -----------------------------------------------------------------------------
# 1. 读取数据与定义变量类别
# -----------------------------------------------------------------------------
file_path = "train_data.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# 定义分类变量与连续变量
onehot_cols = ["SEX", "DEPT", "DIAGNOSIS"]
ordinal_cols = ["TPPA"]
num_cols = ["AGE", "TP", "HIV", "WBC", "RBC", "PLT", "NC", "LY", "NLR"]

# 构建二分类目标变量 (TRUST >= 16 为 1，否则为 0)
X = df.drop(columns=["TRUST"])
y = (df["TRUST"] >= 16).astype(int)

# 划分训练集与测试集 (分层采样以保持二分类比例)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------------------
# 2. 预处理流水线构建
# -----------------------------------------------------------------------------
# 连续变量中位数填充
num_transformer = SimpleImputer(strategy="median")

# 独热编码 (适应新版 sklearn 的 sparse_output 参数)
onehot_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# 序数编码 (处理未见的类别值)
ordinal_transformer = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)

# 组合列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("onehot", onehot_transformer, onehot_cols),
        ("ordinal", ordinal_transformer, ordinal_cols),
    ]
)

# -----------------------------------------------------------------------------
# 3. 构建 SMOTE + 随机森林建模流水线
# -----------------------------------------------------------------------------
# 注意：使用 imblearn.pipeline 确保 SMOTE 仅在训练折内执行
model_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# -----------------------------------------------------------------------------
# 4. 超参数调优 (GridSearchCV)
# -----------------------------------------------------------------------------
# 参数映射：'平衡' 在 sklearn 中对应 'balanced'
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1],
    "classifier__class_weight": ["balanced"],
}

# 实例化网格搜索，不使用多进程 (n_jobs=1)
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",  # 不平衡分类优先使用 ROC-AUC 进行模型选择
    n_jobs=1,
)

# 模型训练
grid_search.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 5. 模型评估
# -----------------------------------------------------------------------------
best_model = grid_search.best_estimator_

# 预测标签与预测概率
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 计算各评估指标
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# 打印评估结果
print("=" * 45)
print(" 随机森林模型评估结果 (TRUST >= 16 预测)")
print("=" * 45)
print(f"最佳超参数配置 : {grid_search.best_params_}")
print("-" * 45)
print(f"准确率 (Accuracy) : {acc:.4f}")
print(f"召回率 (Recall)   : {rec:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"F1分数 (F1-score) : {f1:.4f}")
print(f"ROC-AUC 面积      : {auc:.4f}")
print("=" * 45)