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

# 导入 imblearn 用于处理样本不平衡
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # 1. 读取数据集
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构造目标变量：TRUST 滴度 >= 16 为阳性(1)，否则为阴性(0)
    target_col = df.columns[-1]  # 获取最后一列 TRUST
    y = (df[target_col] >= 16).astype(int)
    X = df.drop(columns=[target_col])

    # 3. 明确特征类型分组
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 构建数据预处理 ColumnTransformer
    # - 连续变量：中位数填充 (Median Imputation)
    # - 分类特征：SEX/DEPT/DIAGNOSIS 采用独热编码，TPPA 采用序数编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
            ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ],
        remainder='drop'
    )

    # 5. 划分训练集与测试集 (80% 训练, 20% 测试，分层抽样保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 使用 imblearn Pipeline 整合预处理、SMOTE 与随机森林
    # 将 SMOTE 放入 Pipeline 可以确保仅在交叉验证的训练折中做过采样，防止验证折数据渗漏
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 定义网格搜索超参数范围
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 8. 超参数调优 (n_jobs=1，不开启多进程)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,
        verbose=1
    )

    print("开始模型训练与超参数调优...")
    grid_search.fit(X_train, y_train)

    # 9. 在独立的测试集上评估最优模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 10. 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 输出结果
    print("\n" + "="*45)
    print(" 最佳超参数组合:")
    for param_name, param_value in grid_search.best_params_.items():
        print(f"  {param_name}: {param_value}")
    print("="*45)
    print(" 测试集评估指标:")
    print(f"  准确率 (Accuracy) : {acc:.4f}")
    print(f"  召回率 (Recall)   : {rec:.4f}")
    print(f"  精确率 (Precision): {prec:.4f}")
    print(f"  F1 分数 (F1-score): {f1:.4f}")
    print(f"  AUC 面积 (ROC-AUC) : {auc:.4f}")
    print("="*45)


if __name__ == '__main__':
    main()