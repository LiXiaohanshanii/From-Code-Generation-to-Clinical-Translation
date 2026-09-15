import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def build_and_evaluate_rf():
    # 1. 加载数据集
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 处理目标列 TRUST：二分类任务 (TRUST >= 16 为 1，否则为 0)
    df['target'] = (df['TRUST'] >= 16).astype(int)

    # 分离特征矩阵与目标变量
    X = df.drop(columns=['TRUST', 'target'])
    y = df['target']

    # 3. 明确特征分组
    onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['TPPA']
    continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 构建预处理管道 (ColumnTransformer)
    # - 连续变量：中位数填充
    # - 分类变量：One-Hot 编码 (SEX, DEPT, DIAGNOSIS)
    # - 分类变量：序数编码 (TPPA)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), continuous_features),
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
            ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
        ],
        remainder='drop'
    )

    # 5. 划分训练集与测试集 (80% 训练，20% 测试，分层抽样保持正负样本比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建包含预处理、SMOTE过采样及随机森林模型的 ImbPipeline
    # 注意：使用 imblearn 的 Pipeline 可确保 SMOTE 仅在交叉验证的训练折中实施，避免数据泄漏
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 7. 超参数网格配置（单进程 n_jobs=1，'平衡' 对应 sklearn 中 'balanced'）
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    # 8. 网格搜索与模型训练
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # 使用 F1 分数指导最佳参数选择
        n_jobs=1  # 不使用多进程
    )

    grid_search.fit(X_train, y_train)

    # 9. 模型评估（在独立的测试集上进行验证）
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各项评价指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 10. 输出结果
    print("=" * 40)
    print("最佳模型参数组合:")
    for param_name, param_value in grid_search.best_params_.items():
        print(f"  {param_name}: {param_value}")
    print("=" * 40)
    print("测试集评估结果:")
    print(f"  准确率 (Accuracy):  {acc:.4f}")
    print(f"  召回率 (Recall):    {rec:.4f}")
    print(f"  精确率 (Precision): {prec:.4f}")
    print(f"  F1 分数 (F1-Score):  {f1:.4f}")
    print(f"  ROC AUC 面积:       {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    build_and_evaluate_rf()