# -*- coding: utf-8 -*-
"""
构建并评估随机森林分类模型（二分类：TRUST >= 16）
数据预处理：中位数填充连续变量缺失值，独热编码/序数编码分类变量，
          SMOTE处理类别不平衡，GridSearchCV调优超参数（n_jobs=1）
评估指标：Accuracy, Recall, Precision, F1-score, AUC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 设置随机种子以保证可重复性
RANDOM_STATE = 42


def main():
    # 1. 加载数据
    data = pd.read_csv('train_data.csv', encoding='utf-8')

    # 2. 分离特征和目标（最后一列为 TRUST）
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 目标二分类：是否 >= 16
    y_bin = (y >= 16).astype(int)

    # 3. 定义特征类型
    categorical_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
    categorical_cols_ordinal = ['TPPA']  # 序数编码
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC',
                       'PLT', 'NC', 'LY', 'NLR']

    # 4. 划分训练集和测试集（分层抽样保证类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.2, stratify=y_bin, random_state=RANDOM_STATE
    )

    # 5. 连续变量缺失值用中位数填充（基于训练集）
    median_vals = X_train[continuous_cols].median()
    X_train_cont = X_train[continuous_cols].fillna(median_vals)
    X_test_cont = X_test[continuous_cols].fillna(median_vals)

    # 6. 分类变量编码
    # 6.1 独热编码 (SEX, DEPT, DIAGNOSIS)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_ohe = ohe.fit_transform(X_train[categorical_cols_onehot])
    X_test_ohe = ohe.transform(X_test[categorical_cols_onehot])

    # 6.2 序数编码 (TPPA)
    ordinal_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_ord = ordinal_enc.fit_transform(X_train[categorical_cols_ordinal])
    X_test_ord = ordinal_enc.transform(X_test[categorical_cols_ordinal])

    # 7. 合并所有特征（按列拼接）
    X_train_processed = np.hstack([
        X_train_cont.values,
        X_train_ohe,
        X_train_ord
    ])
    X_test_processed = np.hstack([
        X_test_cont.values,
        X_test_ohe,
        X_test_ord
    ])

    # 8. 定义使用 SMOTE 的 Pipeline（在交叉验证内部过采样，避免数据泄露）
    smote = SMOTE(random_state=RANDOM_STATE)
    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    pipeline = ImbPipeline([
        ('smote', smote),
        ('rf', rf)
    ])

    # 9. 超参数网格（不使用多进程）
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    # GridSearchCV，n_jobs=1 禁止多进程
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',  # 以 AUC 作为调优指标
        n_jobs=1,
        verbose=1
    )
    grid_search.fit(X_train_processed, y_train)

    # 10. 最佳模型评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_processed)
    y_proba = best_model.predict_proba(X_test_processed)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 打印结果（保留4位小数）
    print("=" * 50)
    print("最佳参数组合:", grid_search.best_params_)
    print("\n模型评估指标（测试集）:")
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1 分数:           {f1:.4f}")
    print(f"AUC 曲线下面积:    {auc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()