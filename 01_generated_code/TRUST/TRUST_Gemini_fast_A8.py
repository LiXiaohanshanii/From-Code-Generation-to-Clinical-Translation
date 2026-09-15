import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # 1. 加载数据
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构建目标变量与特征集
    # 预测目标：TRUST >= 16 为 1，否则为 0
    df['target'] = (df['TRUST'] >= 16).astype(int)

    # 区分特征与目标列
    feature_cols = ['SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA', 'TP', 'HIV',
                    'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    X = df[feature_cols]
    y = df['target']

    # 3. 定义变量类型
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']
    numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 构建数据预处理流水线 (ColumnTransformer)
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量：One-Hot 编码（使用 handle_unknown='ignore' 处理未出现的类别）
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量：序数编码
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ]
    )

    # 5. 划分训练集与测试集 (分层抽样保持正负样本比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 使用 imblearn Pipeline 整合预处理、SMOTE与模型
    # 注意：使用 imblearn 的 Pipeline 可以在交叉验证时只在训练折应用 SMOTE，防止数据泄漏
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 7. 定义超参数网格
    # 注意：通过网格调优时，针对 class_weight='balanced'（中文“平衡”对应的 Scikit-Learn 关键字是 'balanced'）
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    # 8. 网格搜索调优 (不使用多进程 n_jobs=1)
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1  # 禁用多进程
    )

    print("开始模型超参数调优与训练...")
    grid_search.fit(X_train, y_train)
    print("最佳参数组合:", grid_search.best_params_)

    # 9. 模型评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 输出评估结果
    print("\n----------------- 模型评估结果 -----------------")
    print(f"准确率 (Accuracy)   : {acc:.4f}")
    print(f"召回率 (Recall)     : {rec:.4f}")
    print(f"精确率 (Precision)  : {prec:.4f}")
    print(f"F1 分数 (F1-score)   : {f1:.4f}")
    print(f"ROC AUC 面积 (AUC)  : {auc:.4f}")
    print("------------------------------------------------")


if __name__ == '__main__':
    main()