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
    # 1. 读取数据
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 构建目标变量（TRUST >= 16 为1，否则为0）
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # 3. 定义特征分类
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 构建数据预处理流水线
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量：独热编码与序数编码
    onehot_transformer = OneHotEncoder(
        handle_unknown='ignore', sparse_output=False
    )
    ordinal_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols),
        ]
    )

    # 5. 划分训练集与测试集（按类别比例分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建包含 SMOTE 的机器学习流水线（防止交叉验证过程中的数据泄露）
    pipeline = ImbPipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=1)),
        ]
    )

    # 7. 设置超参数网格（禁用多进程 n_jobs=1）
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced'],  # 对应要求中的 '平衡'
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,  # 不使用多进程
    )

    # 8. 模型训练与调优
    print('正在训练模型与网格搜索调优...')
    grid_search.fit(X_train, y_train)

    # 9. 模型评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 10. 输出结果
    print('\n================ 最佳超参数 ================')
    for param_name, param_value in grid_search.best_params_.items():
        clean_name = param_name.replace('classifier__', '')
        print(f'{clean_name}: {param_value}')

    print('\n================ 测试集评估指标 ================')
    print(f'准确率 (Accuracy) : {acc:.4f}')
    print(f'召回率 (Recall)   : {rec:.4f}')
    print(f'精确率 (Precision): {prec:.4f}')
    print(f'F1分数 (F1-score) : {f1:.4f}')
    print(f'AUC 面积 (ROC-AUC): {auc:.4f}')


if __name__ == '__main__':
    main()