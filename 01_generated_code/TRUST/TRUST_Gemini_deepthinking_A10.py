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

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # 1. 读取数据
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 目标变量构建 (二分类：TRUST >= 16 为 1，否则为 0)
    X = df.drop(columns=['TRUST'])
    y = (df['TRUST'] >= 16).astype(int)

    # 3. 特征分类定义
    categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ordinal = ['TPPA']
    continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 构建数据预处理流水线 (ColumnTransformer)
    # 连续变量：中位数填充
    numeric_transformer = SimpleImputer(strategy='median')

    # 分类变量：独热编码
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量：序数编码
    ordinal_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, continuous_features),
            ('onehot', onehot_transformer, categorical_onehot),
            ('ordinal', ordinal_transformer, categorical_ordinal)
        ]
    )

    # 5. 划分训练集与测试集 (分层抽样以保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建组合管道 (预处理 -> SMOTE -> 随机森林分类器)
    # 使用 imblearn.pipeline 确保 SMOTE 只在训练拆分阶段运行，避免评估泄露
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 超参数网格配置 (映射网格参数至管道中的 classifier 节点)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']  # 'balanced' 对应参数要求中的“平衡”
    }

    # 设置网格搜索，n_jobs=1 (禁用多进程)
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1
    )

    print("开始模型训练与超参数调优...")
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print(f"最佳超参数参数组合: {grid_search.best_params_}")

    # 8. 模型预测与评估
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率

    # 计算各评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 打印评估输出
    print("\n" + "=" * 40)
    print("        模型性能评估结果")
    print("=" * 40)
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"F1 分数 (F1-Score): {f1:.4f}")
    print(f"AUC 值 (ROC AUC)  : {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()