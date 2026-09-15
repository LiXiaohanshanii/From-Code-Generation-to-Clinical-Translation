import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 加载数据
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构建二分类目标变量 (TRUST >= 16 为 1，否则为 0)
    df['target'] = (df['TRUST'] >= 16).astype(int)

    # 3. 区分特征列与目标列
    categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ordinal = ['TPPA']
    continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    feature_cols = categorical_onehot + categorical_ordinal + continuous_features
    X = df[feature_cols]
    y = df['target']

    # 4. 划分训练集与测试集 (分层抽样以保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. 构建数据预处理列转换器 (ColumnTransformer)
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 独热编码变量
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 序数编码变量
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, continuous_features),
            ('cat_onehot', onehot_transformer, categorical_onehot),
            ('cat_ordinal', ordinal_transformer, categorical_ordinal)
        ]
    )

    # 6. 构建 Pipeline (整合预处理、SMOTE过采样、随机森林模型)
    # 注意：为了防止数据泄露，SMOTE必须在训练集折分内部进行，故使用 imblearn 的 Pipeline
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 设置超参数网格
    # class_weight 中 'balanced' 映射为 'balanced' 字符串
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 8. 使用 GridSearchCV 进行超参数调优 (按要求 n_jobs=1，不使用多进程)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    # 运行模型训练
    grid_search.fit(X_train, y_train)

    # 9. 获取最佳模型并预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 10. 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 输出结果
    print("=" * 40)
    print("最佳参数配置组合:", grid_search.best_params_)
    print("=" * 40)
    print(f"准确率 (Accuracy)  : {accuracy:.4f}")
    print(f"召回率 (Recall)    : {recall:.4f}")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC 面积 (ROC-AUC) : {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()