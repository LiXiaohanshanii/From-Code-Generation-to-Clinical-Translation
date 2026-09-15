import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCv, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# 1. 加载数据
file_path = "train_data.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# 2. 构建二分类目标变量（预测 TRUST 是否 >= 16）
y = (df["TRUST"] >= 16).astype(int)

# 提取特征矩阵 X
feature_cols = [
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
    "NLR",
]
X = df[feature_cols]

# 定义特征类型
onehot_cols = ["SEX", "DEPT", "DIAGNOSIS"]
ordinal_cols = ["TPPA"]
num_cols = ["AGE", "TP", "HIV", "WBC", "RBC", "PLT", "NC", "LY", "NLR"]

# 3. 构建预处理流水线 (ColumnTransformer)
# 连续变量：中位数填充
num_transformer = SimpleImputer(strategy="median")

# 分类变量编码
onehot_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ordinal_transformer = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("onehot", onehot_transformer, onehot_cols),
        ("ordinal", ordinal_transformer, ordinal_cols),
    ]
)

# 4. 划分为训练集和测试集（80% 训练，20% 测试，分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 构建包含预处理、SMOTE 和 随机森林分类器的管道
# 注意：必须使用 imblearn.pipeline 以确保 SMOTE 仅作用于训练折，避免数据泄露
pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# 6. 配置网格搜索参数（单进程：n_jobs=1）
# 'class_weight': ['balanced'] 对应中文要求 'class_weight': ['平衡']
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1],
    "classifier__class_weight": ["balanced"],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=1,  # 不使用多进程
)

# 拟合模型
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 7. 模型预测与评估
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 8. 打印结果
print("Best Hyperparameters:", grid_search.best_params_)
print("\n--- Model Evaluation Results (Test Set) ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")