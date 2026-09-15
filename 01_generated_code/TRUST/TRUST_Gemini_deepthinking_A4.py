import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# 导入 imblearn 的 Pipeline，保证 SMOTE 仅在训练集折叠中生效，避免数据泄漏
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据集
    file_path = "train_data.csv"
    df = pd.read_csv(file_path, encoding="utf-8")

    # 2. 构建目标变量：TRUST >= 16 为 1，否则为 0
    df["target"] = (df["TRUST"] >= 16).astype(int)

    # 提取特征与目标
    X = df.drop(columns=["TRUST", "target"])
    y = df["target"]

    # 3. 定义变量类型列表
    onehot_features = ["SEX", "DEPT", "DIAGNOSIS"]
    ordinal_features = ["TPPA"]
    continuous_features = [
        "AGE", "TP", "HIV", "WBC", "RBC",
        "PLT", "NC", "LY", "NLR"
    ]

    # 4. 构建预处理流水线 (ColumnTransformer)
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy="median")

    # 分类变量：One-Hot 编码
    onehot_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # 分类变量：序数编码
    ordinal_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, continuous_features),
            ("onehot", onehot_transformer, onehot_features),
            ("ordinal", ordinal_transformer, ordinal_features),
        ]
    )

    # 5. 构建完整建模 Pipeline (预处理 -> SMOTE -> 分类器)
    model_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # 6. 数据集划分 (80% 训练, 20% 测试，保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. 超参数调优设置 (n_jobs=1，不使用多进程)
    # 提示：scikit-learn 中 'class_weight' 的 '平衡' 参数项对应为 'balanced'
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
        verbose=1,
    )

    # 8. 模型训练与调优
    print("开始模型训练与超参数调优...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("\n最佳参数组：", grid_search.best_params_)

    # 9. 模型预测与评估
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 10. 输出评估指标结果
    print("\n" + "=" * 40)
    print("        模型性能评估结果 (Test Set)       ")
    print("=" * 40)
    print(f"准确率 (Accuracy):   {accuracy:.4f}")
    print(f"精准率 (Precision):  {precision:.4f}")
    print(f"召回率 (Recall):     {recall:.4f}")
    print(f"F1 分数 (F1-Score):  {f1:.4f}")
    print(f"AUC 值 (ROC AUC):    {roc_auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()