import pandas as pd
import numpy as np
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
    roc_auc_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 加载数据集
    data_path = 'CrAg_train.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 目标变量转换（二分类任务：CSF-T >= 20 为 1，否则为 0）
    target_col = df.columns[-1]  # 目标列 CSF-T 位于最后一列
    y = (df[target_col] >= 20).astype(int)

    # 特征矩阵
    feature_cols = [
        'SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
        'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS'
    ]
    X = df[feature_cols].copy()

    # 3. 按 4:1 比例划分训练集与验证集（按目标变量分层抽样）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. 定义各变量类型及列名（按题目要求顺序：数值 -> 独热 -> 序数）
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['SER-T', 'Ink staining']

    # 5. 构建数据预处理器 ColumnTransformer
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')
    # 分类变量独热编码
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # 分类变量序数编码
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ]
    )

    # 6. 构建管道（关键点：使用 imblearn.pipeline 防止交叉验证中的数据泄露）
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 定义网格搜索参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 8. 使用 GridSearchCV 进行超参数调优（不使用多进程 n_jobs=1，以 F1 作为评分标准）
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,
        verbose=1
    )

    print("开始在训练集上进行模型训练与超参数调优...")
    grid_search.fit(X_train, y_train)

    print("\n【调优完成】最佳参数组：")
    for param, val in grid_search.best_params_.items():
        print(f"  - {param}: {val}")

    # 9. 在独立的验证集上评估最终模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    # 输出结果
    print("\n================ 验证集模型评估结果 ================")
    print(f"准确率 (Accuracy)       : {acc:.4f}")
    print(f"召回率 (Recall)         : {rec:.4f}")
    print(f"精确率 (Precision)      : {prec:.4f}")
    print(f"F1 分数 (F1-score)      : {f1:.4f}")
    print(f"ROC曲线下面积 (AUC)     : {auc:.4f}")
    print("====================================================")


if __name__ == '__main__':
    main()