import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# 引入 imbalanced-learn 管道以防止交叉验证中的数据泄露
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def build_and_evaluate_rf():
    # 1. 加载数据集
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 2. 构造二分类目标变量 (TRUST >= 16 为 1，否则为 0)
    target_col = 'TRUST'
    X = df.drop(columns=[target_col])
    y = (df[target_col] >= 16).astype(int)

    # 3. 明确定义各类特征列名
    onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['TPPA']
    continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 构建预处理器 (ColumnTransformer)
    # 连续变量中位数填充；SEX、DEPT、DIAGNOSIS 独热编码；TPPA 序数编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), continuous_features),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
        ],
        remainder='drop'
    )

    # 5. 构建完整建模管道 (流水线: 预处理 -> SMOTE -> 随机森林)
    # 在管道内部使用 SMOTE 可以确保过采样仅在训练折内进行，防止测试折信息泄露
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 6. 划分训练集与测试集 (采用分层采样保持正负样本比例一致)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. 配置网格搜索参数范围 (不使用多进程 n_jobs=1)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1
    )

    # 8. 模型训练与参数调优
    grid_search.fit(X_train, y_train)

    # 9. 获取最佳模型并在独立测试集上进行评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 10. 输出评估结果
    print("=" * 50)
    print("最佳超参数配置组合：")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    print("模型在测试集上的评估指标：")
    print(f"  准确率 (Accuracy)  : {acc:.4f}")
    print(f"  召回率 (Recall)    : {rec:.4f}")
    print(f"  精确率 (Precision) : {prec:.4f}")
    print(f"  F1分数 (F1-score)  : {f1:.4f}")
    print(f"  ROC-AUC 面积       : {auc:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    build_and_evaluate_rf()