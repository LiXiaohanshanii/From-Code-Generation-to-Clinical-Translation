import numpy as np
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# -------------------------------------------------------------------------
# 1. 数据加载与二分类目标转化
# -------------------------------------------------------------------------
# 加载数据集
file_path = "CrAg_train.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# 假设目标列 CSF-T 位于最后一列
target_col = df.columns[-1]

# 将目标列构建为二分类任务（≥20 为 1，<20 为 0）
y = (df[target_col] >= 20).astype(int)
X = df.drop(columns=[target_col])

# -------------------------------------------------------------------------
# 2. 定义特征类型分类
# -------------------------------------------------------------------------
# 连续变量
num_features = ["CL", "GLU", "Protein", "RBC", "AGE"]

# 序数编码变量 (SER-T, Ink staining)
ordinal_features = ["SER-T", "Ink staining"]

# 独热编码变量 (Color, Transparency, SEX, DEPT, DIAGNOSIS)
onehot_features = ["Color", "Transparency", "SEX", "DEPT", "DIAGNOSIS"]

# -------------------------------------------------------------------------
# 3. 构建预处理 Pipeline (Preprocessors)
# -------------------------------------------------------------------------
# 连续变量预处理：中位数填充缺失值
num_transformer = SimpleImputer(strategy="median")

# 序数编码预处理：处理潜在未见的分类标签
ordinal_transformer = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)

# 独热编码预处理：忽略未见过的分类标签
onehot_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# 使用 ColumnTransformer 整合特征预处理
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("ord", ordinal_transformer, ordinal_features),
        ("cat", onehot_transformer, onehot_features),
    ]
)

# -------------------------------------------------------------------------
# 4. 构建包含 SMOTE 和 随机森林 的完整 Pipeline
# -------------------------------------------------------------------------
# 注意：为了防止数据泄露（Data Leakage），SMOTE 必须使用 imblearn 的 Pipeline 进行封装
model_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# -------------------------------------------------------------------------
# 5. 数据集划分 (Train-Test Split)
# -------------------------------------------------------------------------
# 使用分层抽样 (stratify=y) 保证训练集和测试集的正负样本比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------------------------
# 6. 超参数调优 (GridSearchCV)
# -------------------------------------------------------------------------
# 针对 ImbPipeline，模型参数需指定前缀 'classifier__'
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1],
    "classifier__class_weight": ["balanced"],
}

# n_jobs=1 严格遵守“不使用多进程”要求
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=1,
    verbose=1,
)

# 拟合模型
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# -------------------------------------------------------------------------
# 7. 模型评估与结果输出
# -------------------------------------------------------------------------
# 在独立测试集上预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)

# 格式化打印评估报告
print("\n" + "=" * 50)
print(" 最佳模型超参数选择: ")
print(grid_search.best_params_)
print("=" * 50)
print(" 测试集评估结果 (Evaluation Metrics): ")
print(f"  - 准确率 (Accuracy)  : {accuracy:.4f}")
print(f"  - 精确率 (Precision) : {precision:.4f}")
print(f"  - 召回率 (Recall)    : {recall:.4f}")
print(f"  - F1 分数 (F1-score) : {f1:.4f}")
print(f"  - AUC 值 (ROC-AUC)   : {auc:.4f}")
print("=" * 50)