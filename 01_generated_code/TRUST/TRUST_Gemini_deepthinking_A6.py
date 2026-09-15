import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# 使用 imblearn 的 Pipeline 避免 SMOTE 在交叉验证中产生数据泄漏
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 定义特征分类与目标变量转换
    cat_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ordinal = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 构建特征矩阵 X 与 目标变量 y (TRUST >= 16 为 1，否则为 0)
    X = df[cat_onehot + cat_ordinal + num_cols]
    y = (df['TRUST'] >= 16).astype(int)

    # 3. 划分训练集与测试集 (分层抽样保持正负样本比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 构建预处理器 (ColumnTransformer)
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量：独热编码
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量：序数编码
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat_onehot', onehot_transformer, cat_onehot),
            ('cat_ordinal', ordinal_transformer, cat_ordinal)
        ]
    )

    # 5. 构建整合 SMOTE 与 随机森林的 Pipeline
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 6. 配置超参数网格 (通过 classifier__ 前缀指向模型参数)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']  # '平衡' 对应 sklearn 中的 'balanced'
    }

    # 7. 网格搜索 (单进程 n_jobs=1)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',  # 采用 AUC 作为超参数筛选标准
        n_jobs=1  # 禁用多进程
    )

    print("开始模型训练与超参数调优...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("\n最佳超参数组合：")
    for param, value in grid_search.best_params_.items():
        print(f"  {param.split('__')[1]}: {value}")

    # 8. 在测试集上评估模型性能
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 9. 输出评估指标
    print("\n" + "=" * 35)
    print("      模型评估结果 (Test Set)")
    print("=" * 35)
    print(f" 准确率 (Accuracy)         : {accuracy:.4f}")
    print(f" 召回率 (Recall)           : {recall:.4f}")
    print(f" 精确率 (Precision)        : {precision:.4f}")
    print(f" F1 分数 (F1-score)        : {f1:.4f}")
    print(f" ROC 曲线下面积 (AUC)       : {auc:.4f}")
    print("=" * 35)


if __name__ == '__main__':
    main()