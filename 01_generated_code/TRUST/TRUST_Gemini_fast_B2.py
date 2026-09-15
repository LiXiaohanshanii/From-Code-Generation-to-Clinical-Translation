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


def main():
    # 1. 读取数据集
    file_path = "train_data.csv"
    df = pd.read_csv(file_path, encoding="utf-8")

    # 处理潜在的特征列名拼写差异（如 DIAGNOSIS 与 DIAGONSIS）
    if "DIAGONSIS" in df.columns and "DIAGNOSIS" not in df.columns:
        df.rename(columns={"DIAGONSIS": "DIAGNOSIS"}, inplace=True)

    # 2. 构建目标变量：TRUST >= 16 记为 1，否则记为 0
    df["TARGET"] = (df["TRUST"] >= 16).astype(int)

    # 提取特征集 X 与 目标集 y
    X = df.drop(columns=["TRUST", "TARGET"])
    y = df["TARGET"]

    # 3. 区分列属性
    onehot_cols = ["SEX", "DEPT", "DIAGNOSIS"]
    ordinal_cols = ["TPPA"]
    num_cols = ["AGE", "TP", "HIV", "WBC", "RBC", "PLT", "NC", "LY", "NLR"]

    # 4. 构建预处理流水线 (ColumnTransformer)
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy="median")

    # 分类变量：独热编码
    onehot_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    # 序数分类变量：序数编码
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

    # 5. 划分训练集与测试集 (分层抽样保持正负样本比例一致)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建包含 SMOTE 和 随机森林 的组合流水线
    # 使用 imblearn.pipeline 可以避免 SMOTE 在交叉验证时造成测试集信息泄漏
    full_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # 7. 网格搜索超参数调优 (按要求禁用多进程 n_jobs=1)
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10],
        "classifier__min_samples_split": [2],
        "classifier__min_samples_leaf": [1],
        "classifier__class_weight": ["balanced"],  # 对应中文“平衡”配置
    }

    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",  # 针对不平衡数据，优先使用 F1 评估网格模型
        n_jobs=1,  # 不使用多进程
    )

    # 执行模型训练与参数搜索
    grid_search.fit(X_train, y_train)

    # 8. 获取最优模型并对测试集进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率值

    # 9. 计算并输出各项模型评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("=" * 40)
    print("网格搜索最佳参数配置:")
    for param_name, param_value in grid_search.best_params_.items():
        print(f"  {param_name}: {param_value}")

    print("=" * 40)
    print("模型评估结果（测试集）:")
    print(f"  准确率 (Accuracy)   : {acc:.4f}")
    print(f"  召回率 (Recall)     : {rec:.4f}")
    print(f"  精确率 (Precision)  : {prec:.4f}")
    print(f"  F1分数 (F1-score)   : {f1:.4f}")
    print(f"  ROC-AUC 面积 (AUC)  : {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()