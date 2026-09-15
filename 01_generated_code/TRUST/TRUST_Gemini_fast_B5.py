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
    roc_auc_score,
    classification_report
)
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据集
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构建目标变量（二分类任务：TRUST >= 16 标记为 1，否则为 0）
    df['target'] = (df['TRUST'] >= 16).astype(int)

    # 划分特征矩阵 X 与目标向量 y（移除原始 TRUST 列）
    X = df.drop(columns=['TRUST', 'target'])
    y = df['target']

    # 定义变量类型
    categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ordinal = ['TPPA']
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 3. 划分训练集和测试集（分层抽样保持类分布一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 构建预处理流水线 (ColumnTransformer)
    # - 连续变量：中位数填充
    # - SEX、DEPT、DIAGNOSIS：独热编码
    # - TPPA：序数编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), continuous_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_onehot),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_ordinal)
        ]
    )

    # 5. 对特征数据进行拟合与转换
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # 6. 处理类别不平衡：仅对训练集使用 SMOTE 过采样
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

    # 7. 定义随机森林分类器及超参数网格
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']  # 对应参数要求中的“平衡”
    }

    # 8. 网格搜索调优（使用单进程 n_jobs=1）
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1
    )

    grid_search.fit(X_train_resampled, y_train_resampled)

    # 输出最佳参数
    best_model = grid_search.best_estimator_
    print("================ 调优结果 ================")
    print(f"最佳超参数组合: {grid_search.best_params_}\n")

    # 9. 模型评估（在独立的测试集上评估）
    y_pred = best_model.predict(X_test_encoded)
    y_pred_proba = best_model.predict_proba(X_test_encoded)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 10. 打印评估指标结果
    print("================ 测试集模型评估指标 ================")
    print(f"准确率 (Accuracy):   {accuracy:.4f}")
    print(f"召回率 (Recall):     {recall:.4f}")
    print(f"精确率 (Precision):  {precision:.4f}")
    print(f"F1分数 (F1-Score):   {f1:.4f}")
    print(f"ROC-AUC 曲线面积:    {auc:.4f}")
    print("==================================================")


if __name__ == '__main__':
    main()