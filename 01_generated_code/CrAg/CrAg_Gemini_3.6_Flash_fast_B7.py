import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

# 确保 imblearn 库已安装 (若未安装可通过 `pip install imbalanced-learn` 安装)
from imblearn.over_sampling import SMOTE


def main():
    # ---------------------------------------------------------
    # 1. 读取数据
    # ---------------------------------------------------------
    file_path = "CrAg_train.csv"
    data = pd.read_csv(file_path, encoding="utf-8")

    # ---------------------------------------------------------
    # 2. 构建目标变量 (二分类: CSF-T >= 20 为 1，否则为 0)
    # ---------------------------------------------------------
    # 假设目标列名称为 'CSF-T'（数据集最后一列）
    target_col = "CSF-T"

    # 构建二分类标签
    y = (data[target_col] >= 20).astype(int)
    X = data.drop(columns=[target_col])

    # ---------------------------------------------------------
    # 3. 特征分类与数据预处理
    # ---------------------------------------------------------
    # 连续变量
    num_cols = ["CL", "GLU", "Protein", "RBC", "AGE"]
    # 序数编码分类变量
    ordinal_cols = ["SER-T", "Ink staining"]
    # 独热编码分类变量
    onehot_cols = ["Color", "Transparency", "SEX", "DEPT", "DIAGNOSIS"]

    # 连续变量：中位数填充
    num_imputer = SimpleImputer(strategy="median")
    X_num = num_imputer.fit_transform(X[num_cols])

    # 序数编码（分类变量无缺失值）
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    X_ordinal = ordinal_encoder.fit_transform(X[ordinal_cols])

    # 独热编码（分类变量无缺失值）
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_onehot = onehot_encoder.fit_transform(X[onehot_cols])

    # 拼接预处理后的特征矩阵
    X_processed = np.hstack([X_num, X_ordinal, X_onehot])

    # ---------------------------------------------------------
    # 4. 数据集拆分与 SMOTE 不平衡处理
    # ---------------------------------------------------------
    # 拆分训练集与测试集（保持类别比例分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # 仅在训练集上应用 SMOTE，防止数据泄漏 (Data Leakage)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # ---------------------------------------------------------
    # 5. 模型构建与网格搜索超参数调优
    # ---------------------------------------------------------
    rf = RandomForestClassifier(random_state=42)

    # 按要求设置超参数网格
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "class_weight": ["balanced"],
    }

    # 使用 GridSearchCV 进行调优，不用多进程 (n_jobs=1)，以 F1 作为评分指标
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train_resampled, y_train_resampled)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # ---------------------------------------------------------
    # 6. 模型评估
    # ---------------------------------------------------------
    # 预测类别与概率
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各项评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估结果
    print("\n================ 模型评估结果 ================")
    print(f"最佳超参数组合: {grid_search.best_params_}")
    print(f"准确率 (Accuracy):   {acc:.4f}")
    print(f"精确率 (Precision):  {prec:.4f}")
    print(f"召回率 (Recall):     {rec:.4f}")
    print(f"F1 分数 (F1-Score):  {f1:.4f}")
    print(f"ROC-AUC 面积 (AUC):  {auc:.4f}")
    print("==============================================")


if __name__ == "__main__":
    main()