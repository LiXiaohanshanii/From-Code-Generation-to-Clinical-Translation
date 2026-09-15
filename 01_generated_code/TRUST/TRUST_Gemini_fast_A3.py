import pandas as pd
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
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 构建目标变量（判断 TRUST 是否 >= 16，转为二分类任务 0 和 1）
    y = (df['TRUST'] >= 16).astype(int)

    # 3. 区分特征列
    feature_cols = [
        'SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA',
        'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR'
    ]
    X = df[feature_cols]

    # 定义不同类型的特征列表
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']
    numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 数据集拆分 (保持训练集与测试集分布一致)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. 定义特征预处理 Pipeline
    # 连续变量：中位数填充
    numeric_transformer = SimpleImputer(strategy='median')

    # 分类变量：One-Hot 编码 (handle_unknown='ignore' 避免测试集中出现未见类别报错)
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量：序数编码
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 使用 ColumnTransformer 组装预处理模块
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ]
    )

    # 6. 构建包含预处理、SMOTE 和分类器的完整 Pipeline
    # 注意：必须使用 imblearn.pipeline 才能保证 SMOTE 只应用于训练折/训练集
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 超参数网格配置 (注意参数名前缀需与 Pipeline 中定义一致)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 8. 网格搜索 (使用单进程 n_jobs=1，以 ROC_AUC 为评估指标选择最佳模型)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=1,
        verbose=1
    )

    # 模型训练与调优
    grid_search.fit(X_train, y_train)

    # 9. 模型评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 10. 输出评估结果
    print("\n" + "=" * 40)
    print("最佳超参数组合:")
    print(grid_search.best_params_)
    print("=" * 40)
    print("测试集评估指标结果:")
    print(f"准确率 (Accuracy) : {accuracy:.4f}")
    print(f"召回率 (Recall)   : {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1 分数 (F1-score) : {f1:.4f}")
    print(f"AUC 面积 (ROC AUC) : {roc_auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()