import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)

# 使用 imblearn 的 Pipeline，防止网格搜索过采样时发生数据泄漏
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 加载数据
    data_path = "train_data.csv"
    df = pd.read_csv(data_path, encoding="utf-8")

    # 2. 构建目标变量（二分类：TRUST >= 16 为 1，否则为 0）
    # 转换为数值型，防止可能存在的异常字符
    y_raw = pd.to_numeric(df["TRUST"], errors="coerce")
    y = (y_raw >= 16).astype(int)

    # 特征矩阵
    X = df.drop(columns=["TRUST"])

    # 定义特征列分类
    one_hot_cols = ["SEX", "DEPT", "DIAGNOSIS"]
    ordinal_cols = ["TPPA"]
    num_cols = ["AGE", "TP", "HIV", "WBC", "RBC", "PLT", "NC", "LY", "NLR"]

    # 3. 构建预处理 ColumnTransformer
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy="median")

    # 分类变量（独热编码）：未知类别自动忽略
    one_hot_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # 分类变量（序数编码）：未知类别自动填充为 -1
    ordinal_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat_onehot", one_hot_transformer, one_hot_cols),
            ("cat_ordinal", ordinal_transformer, ordinal_cols),
        ]
    )

    # 4. 构建包含预处理、SMOTE 和分类器的 Pipeline
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # 5. 划分数据集（分层抽样，保证训练集和测试集标签比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 超参数调优设置
    # 网格搜索的参数字典（注意管道中分类器的名前缀 'classifier__'）
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],  # 映射中文‘平衡’为 sklearn 标准参数
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",  # 针对类别不平衡，优选 F1 评价搜索
        cv=5,
        n_jobs=1,  # 不使用多进程
    )

    # 训练模型并执行超参数调优
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 7. 模型预测与评估
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 输出结果
    print("=== 网格搜索最佳参数 ===")
    print(grid_search.best_params_)
    print("\n=== 模型测试集评估结果 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"召回率 (Recall)  : {recall:.4f}")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"F1 分数 (F1-score) : {f1:.4f}")
    print(f"ROC-AUC 面积     : {auc:.4f}")


if __name__ == "__main__":
    main()