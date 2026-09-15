# -*- coding: utf-8 -*-
"""
CrAg (隐球菌抗原) 滴度预测模型 - 随机森林二分类
适用环境: PyCharm 2025.2.3 / Python 3.10+
作者: 检验科数据分析组
描述: 基于CSF常规及生化指标预测CrAg滴度是否≥20
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath: str):
    """加载数据并进行目标变量二值化"""
    df = pd.read_csv(filepath, encoding='utf-8')

    # 定义特征列与目标列
    feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                    'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
    target_col = 'CSF-T'

    X = df[feature_cols].copy()
    y = (df[target_col] >= 20).astype(int)  # 二分类: ≥20为1, <20为0

    print(f"数据集形状: {X.shape}")
    print(f"目标变量分布:\n{y.value_counts()}")
    print(f"阳性(≥20)比例: {y.mean():.2%}")

    return X, y


def build_preprocessor():
    """
    构建数据预处理管道
    - 连续变量: 中位数填充
    - 独热编码: Color, Transparency, SEX, DEPT, DIAGNOSIS
    - 序数编码: SER-T, Ink staining
    """
    # 连续变量
    numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    numeric_transformer = Pipeline(steps=[
        ('median_imputer', SimpleImputer(strategy='median'))
    ])

    # 独热编码变量
    ohe_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ohe_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 序数编码变量 (按临床意义排序)
    ordinal_features = ['SER-T', 'Ink staining']
    # 根据实际数据中的类别自动推断顺序；若需固定顺序可手动指定categories参数
    ord_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ohe', ohe_transformer, ohe_features),
            ('ord', ord_transformer, ordinal_features)
        ],
        remainder='drop'  # 丢弃未指定的列，防止数据泄露
    )

    return preprocessor


def train_and_evaluate(X, y):
    """训练随机森林模型并输出评估指标"""

    # 划分训练集与测试集 (分层抽样保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 构建包含SMOTE的完整管道
    # SMOTE仅在训练集上执行，避免测试集数据泄露
    pipeline = ImbPipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 超参数网格 (注意: ImbPipeline中参数名需加classifier__前缀)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 网格搜索 (不使用多进程 n_jobs=1)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,
        verbose=1
    )

    print("\n开始超参数调优...")
    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1: {grid_search.best_score_:.4f}")

    # 在测试集上评估
    y_pred = grid_search.predict(X_test)
    y_prob = grid_search.predict_proba(X_test)[:, 1]

    # 计算各项指标
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }

    print("\n===== 测试集评估结果 =====")
    for name, value in metrics.items():
        print(f"{name:>12s}: {value:.4f}")

    return grid_search, metrics


if __name__ == '__main__':
    # ========== 主程序入口 ==========
    DATA_PATH = 'CrAg_train.csv'

    X, y = load_and_prepare_data(DATA_PATH)
    best_model, eval_metrics = train_and_evaluate(X, y)

    # 如需保存模型供后续临床验证使用，可取消下方注释
    # import joblib
    # joblib.dump(best_model, 'CrAg_RF_model.pkl')
    # print("\n模型已保存至 CrAg_RF_model.pkl")