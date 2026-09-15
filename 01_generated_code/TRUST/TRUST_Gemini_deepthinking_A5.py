import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# 使用 imblearn 提供的 Pipeline，防止在交叉验证时产生数据泄露
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构建目标变量与特征集 (二分类：TRUST >= 16 为 1，否则为 0)
    # 处理可能的非数值数据类型
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # 定义变量类型
    onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['TPPA']
    numeric_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 3. 预处理流水线搭建
    # 连续变量中位数填充
    numeric_transformer = SimpleImputer(strategy='median')

    # 分类变量独热编码
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量序数编码
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 整合特征预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('onehot', onehot_transformer, onehot_features),
            ('ordinal', ordinal_transformer, ordinal_features)
        ]
    )

    # 4. 划分训练集与测试集（按照 8:2 比例，并保持类别分层比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. 构建包含预处理、SMOTE采样和随机森林分类器的集成流水线
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 6. 超参数调优网格配置
    # 注意：'class_weight': ['平衡'] 在 scikit-learn 中对应参数值 'balanced'
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 7. 网格搜索超参数调优（不使用多进程 n_jobs=1）
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=1
    )

    print("开始模型超参数调优与训练...")
    grid_search.fit(X_train, y_train)

    print("\n最佳超参数组合：")
    for param, value in grid_search.best_params_.items():
        print(f"  {param.split('__')[1]}: {value}")

    # 8. 获取最佳模型并在测试集上进行评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 9. 输出评估结果
    print("\n================ 测试集评估指标 ================")
    print(f"准确率 (Accuracy)       : {acc:.4f}")
    print(f"召回率 (Recall)         : {rec:.4f}")
    print(f"精确率 (Precision)      : {prec:.4f}")
    print(f"F1 分数 (F1-score)      : {f1:.4f}")
    print(f"ROC 曲线下面积 (AUC)    : {auc:.4f}")
    print("================================================")


if __name__ == '__main__':
    main()