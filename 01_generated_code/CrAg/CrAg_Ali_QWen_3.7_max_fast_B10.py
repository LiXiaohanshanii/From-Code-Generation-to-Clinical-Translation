"""
CrAg (隐球菌抗原) 滴度预测模型 - 随机森林二分类
适用环境: PyCharm 2025.2.3 / Python 3.9+
作者: 检验科AI辅助诊断小组
描述: 预测CSF-T滴度是否≥20，用于隐球菌脑膜炎快速筛查辅助
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 抑制无关警告信息
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath: str):
    """加载数据并进行目标变量二值化转换"""
    df = pd.read_csv(filepath, encoding='utf-8')

    # 定义特征列与目标列
    feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                    'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
    target_col = 'CSF-T'

    X = df[feature_cols].copy()
    y = (df[target_col] >= 20).astype(int)  # 二分类: ≥20为阳性(1), <20为阴性(0)

    print(f"数据集形状: {X.shape}")
    print(f"目标变量分布:\n{y.value_counts()}")
    print(f"阳性比例: {y.mean():.2%}")

    return X, y


def build_preprocessor():
    """
    构建预处理管道:
    - 连续变量: 中位数填充缺失值
    - 独热编码变量: Color, Transparency, SEX, DEPT, DIAGNOSIS
    - 序数编码变量: SER-T, Ink staining
    """
    # 根据数据类型自动识别，避免硬编码列名顺序问题
    continuous_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['SER-T', 'Ink staining']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), continuous_features),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
        ],
        remainder='drop'  # 丢弃未指定的列，保证安全
    )

    return preprocessor


def train_and_evaluate(X, y):
    """训练随机森林模型并输出评估指标"""

    # 划分训练集与测试集 (分层抽样保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 构建包含SMOTE的完整管道
    # SMOTE仅在训练集上执行，防止数据泄露到测试集
    pipeline = ImbPipeline([
        ('preprocessor', build_preprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 超参数网格搜索 (不使用多进程 n_jobs=1)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,       # 明确禁用多进程
        verbose=1,
        refit=True
    )

    print("\n===== 开始超参数调优 =====")
    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1: {grid_search.best_score_:.4f}")

    # 在独立测试集上评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 计算各项评估指标
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }

    print("\n===== 测试集评估结果 =====")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:<12}: {value:.4f}")

    return best_model, metrics


if __name__ == '__main__':
    # ========== 主程序入口 ==========
    DATA_PATH = 'CrAg_train.csv'

    # 1. 加载与准备数据
    X, y = load_and_prepare_data(DATA_PATH)

    # 2. 训练与评估
    model, evaluation_metrics = train_and_evaluate(X, y)

    print("\n✅ 模型构建与评估完成")