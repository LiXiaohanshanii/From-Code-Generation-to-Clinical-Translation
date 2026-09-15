import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def main():
    # 1. 加载数据集
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构造目标变量（二分类：TRUST >= 16 为 1，否则为 0）
    X = df.drop(columns=['TRUST'])
    y = (df['TRUST'] >= 16).astype(int)

    # 3. 定义特征分类
    onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['TPPA']
    continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 构建预处理 ColumnTransformer
    # - 连续变量：中位数填充
    # - 独热编码变量：OneHotEncoder (handle_unknown='ignore' 避免测试集遇到未见类别报错)
    # - 序数编码变量：OrdinalEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), continuous_features),
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
            ('cat_ordinal', OrdinalEncoder(), ordinal_features)
        ]
    )

    # 5. 构建 imblearn Pipeline
    # 顺序：预处理 -> SMOTE过采样 -> 随机森林分类器
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 6. 数据集划分 (按8:2划分训练集与测试集)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. 定义超参数网格并配置 GridSearchCV
    # 注意：网格参数中将 'balanced' 映射为字符串 'balanced'
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 设 n_jobs=1 (不用多进程)
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    # 8. 模型训练与调优
    print("开始进行模型训练与超参数调优...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳超参数配置: {grid_search.best_params_}\n")

    # 9. 模型评估（基于测试集）
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 10. 输出评估指标结果
    print("----- 模型评估结果 (Test Set) -----")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 分数 (F1-score): {f1:.4f}")
    print(f"AUC 曲线下面积 (ROC AUC): {auc:.4f}")

if __name__ == '__main__':
    main()