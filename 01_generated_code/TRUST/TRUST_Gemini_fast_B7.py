import os
import numpy as np
import pandas as pd

# 导入机器学习与数据预处理模块
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# 导入 imblearn 中的 SMOTE 和 Pipeline，防止交叉验证中的数据泄露
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # -------------------------------------------------------------------------
    # 1. 数据加载与二分类标签转换
    # -------------------------------------------------------------------------
    file_path = 'train_data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据集文件: {file_path}，请确保该文件与代码位于同一目录下。")

    # 读取 CSV 文件，指定 utf-8 编码
    df = pd.read_csv(file_path, encoding='utf-8')

    # 定义列分类
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码变量
    ordinal_cols = ['TPPA']  # 序数编码变量

    feature_cols = onehot_cols + ordinal_cols + num_cols
    target_col = 'TRUST'

    # 检查特征列与目标列是否存在
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise KeyError(f"数据集中缺失以下列: {missing_cols}")

    # 特征矩阵 X 与目标向量 y
    X = df[feature_cols].copy()

    # 任务目标：预测 TRUST 是否 >= 16（转换为 0/1 二分类问题）
    # 注：如果数据中 TRUST 含有字符串（如 "1:16"），请先清洗提取数值
    y = (pd.to_numeric(df[target_col], errors='coerce') >= 16).astype(int)

    # 划分训练集与测试集（按标签比例分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------------------------------------------------
    # 2. 预处理流水线构建
    # -------------------------------------------------------------------------
    # 连续变量：中位数填充缺失值
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量：SEX, DEPT, DIAGNOSIS 进行独热编码
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量：TPPA 进行序数编码
    ordinal_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    # 组合 ColumnTransformer 处理不同类型的列
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ]
    )

    # -------------------------------------------------------------------------
    # 3. 组合预处理、SMOTE 与随机森林模型（Pipeline 管道）
    # -------------------------------------------------------------------------
    # 使用 imblearn 的 Pipeline 可以保证 SMOTE 仅在交叉验证的训练折中执行，避免数据泄露
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # -------------------------------------------------------------------------
    # 4. 超参数调优（GridSearchCV）
    # -------------------------------------------------------------------------
    # 注意：在 scikit-learn 中，'class_weight' 的平衡参数为英文字符串 'balanced'
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 不使用多进程，显式指定 n_jobs=1
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=1
    )

    print("开始模型训练与网格搜索超参数调优...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("\n最佳超参数组合：")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    # -------------------------------------------------------------------------
    # 5. 模型评估
    # -------------------------------------------------------------------------
    # 对测试集进行预测
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估结果
    print("\n" + "=" * 40)
    print("      测试集模型评估结果 (TRUST >= 16)")
    print("=" * 40)
    print(f"准确率 (Accuracy)       : {acc:.4f}")
    print(f"精确率 (Precision)      : {prec:.4f}")
    print(f"召回率 (Recall)         : {rec:.4f}")
    print(f"F1分数 (F1-Score)       : {f1:.4f}")
    print(f"ROC-AUC 面积 (AUC)      : {auc:.4f}")
    print("=" * 40)

    print("\n详细分类报告：")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    main()