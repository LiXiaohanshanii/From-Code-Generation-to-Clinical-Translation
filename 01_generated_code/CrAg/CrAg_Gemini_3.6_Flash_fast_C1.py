import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据集
    file_path = "CrAg_train.csv"
    df = pd.read_csv(file_path, encoding="utf-8")

    # 2. 构建目标变量（CSF-T ≥ 20 为阳性类 1，< 20 为阴性类 0）
    target_col = df.columns[-1]
    df["target"] = (df[target_col] >= 20).astype(int)

    # 3. 明确特征列分组（严格按照：数值 -> 独热 -> 序数 的数据处理顺序）
    num_cols = ["CL", "GLU", "Protein", "RBC", "AGE"]
    ohe_cols = ["Color", "Transparency", "SEX", "DEPT", "DIAGNOSIS"]
    ord_cols = ["SER-T", "Ink staining"]

    X = df[num_cols + ohe_cols + ord_cols]
    y = df["target"]

    # 4. 划分训练集与验证集（比例 4:1，启用分层抽样以保持滴度阳性率一致）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. 构建预处理器 (ColumnTransformer)
    # 连续变量：中位数填充；分类变量：独热编码 / 序数编码
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ohe_cols,
            ),
            (
                "ord",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                ord_cols,
            ),
        ],
        remainder="drop",
    )

    # 6. 构建 imblearn Pipeline（确保 SMOTE 在交叉验证折内进行，严防数据泄露）
    pipeline = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # 7. 超参数网格设置（不使用多进程 n_jobs=1）
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

    # 8. 模型训练与寻优
    print("正在进行模型超参数调优...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳超参数组合: {grid_search.best_params_}")

    # 9. 验证集评估
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    # 10. 输出评估指标
    print("\n========== 验证集模型评估结果 ==========")
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC 曲线下面积    : {auc:.4f}")
    print("==========================================")


if __name__ == "__main__":
    main()