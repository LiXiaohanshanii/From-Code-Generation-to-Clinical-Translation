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
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def run_trust_rf_pipeline(data_path: str = 'train_data.csv'):
    # 1. 加载数据集
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 构造目标变量 y：TRUST 滴度是否 >= 16（二分类任务：1表示高滴度，0表示低滴度/阴性）
    target_col = 'TRUST'
    y = (df[target_col] >= 16).astype(int)

    # 3. 特征分类划分
    continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['TPPA']

    all_feature_cols = onehot_features + ordinal_features + continuous_features
    X = df[all_feature_cols]

    # 4. 划分训练集与测试集（按目标变量分层抽样，保持类比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. 构建数据预处理转换器 (ColumnTransformer)
    # - 连续变量：中位数填充
    # - 独热编码变量：OneHotEncoder
    # - 序数编码变量：OrdinalEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), continuous_features),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
        ]
    )

    # 6. 组合包含预处理、SMOTE过采样与随机森林的机器学习管道
    # 注意：使用 imblearn.pipeline 可以确保 SMOTE 仅作用于 Cross-Validation 的训练折上，防止测试折泄漏
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
    ])

    # 7. 设置超参数搜索网格
    # 注：Scikit-Learn 中 class_weight 的 '平衡' 参数值为 'balanced'
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 8. 构建网格搜索（单进程运行 n_jobs=1）
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    # 执行拟合与超参数调优
    print("正在运行超参数搜索与模型训练...")
    grid_search.fit(X_train, y_train)

    # 9. 模型评估（使用测试集）
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算 5 项核心评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 10. 打印结果
    print("\n" + "=" * 50)
    print("          随机森林模型评估报告          ")
    print("=" * 50)
    print(f"最佳参数组合 : {grid_search.best_params_}")
    print("-" * 50)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}")
    print(f"精确率 (Precision) : {prec:.4f}")
    print(f"F1 分数 (F1-score) : {f1:.4f}")
    print(f"AUC 面积 (ROC-AUC) : {auc:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    run_trust_rf_pipeline('train_data.csv')