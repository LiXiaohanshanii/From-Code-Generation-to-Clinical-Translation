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


def load_and_prepare_data(filepath: str):
    """加载数据并将TRUST滴度转换为二分类标签(>=16为1, <16为0)"""
    df = pd.read_csv(filepath, encoding='utf-8')

    # 定义特征列
    cat_cols = ['SEX', 'DEPT', 'DIAGNOSIS', 'TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    feature_cols = cat_cols + num_cols

    X = df[feature_cols].copy()
    y = (df['TRUST'] >= 16).astype(int)  # 二分类: ≥16 -> 1, <16 -> 0

    print(f"数据集形状: {X.shape}")
    print(f"目标变量分布:\n{y.value_counts()}")
    print(f"阳性(≥16)比例: {y.mean():.4f}")

    return X, y, cat_cols, num_cols


def build_pipeline(cat_cols, num_cols):
    """
    构建包含预处理+SMOTE+随机森林的完整Pipeline
    注意: SMOTE放在Pipeline内部可防止测试集信息泄露
    """
    # TPPA序数编码（按滴度逻辑排序，若实际类别不同请调整categories）
    tppa_categories = [sorted(['阴性', '弱阳性', '阳性', '强阳性'])]
    # 如果TPPA原始值就是数值型有序类别，可改为: tppa_categories = [[1, 2, 4, 8]]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_imputer', SimpleImputer(strategy='median'), num_cols),
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             ['SEX', 'DEPT', 'DIAGNOSIS']),
            ('cat_ordinal', OrdinalEncoder(categories=tppa_categories,
                                           handle_unknown='use_encoded_value',
                                           unknown_value=-1),
             ['TPPA'])
        ],
        remainder='drop'
    )

    # 使用imblearn的Pipeline确保SMOTE仅在训练折上执行
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return pipe


def tune_and_evaluate(X, y, cat_cols, num_cols):
    """超参数调优与模型评估（单进程）"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(cat_cols, num_cols)

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='f1',          # 不平衡数据以F1为优化目标
        cv=5,
        n_jobs=1,              # 不使用多进程
        verbose=1,
        refit=True
    )

    print("\n===== 开始超参数调优 =====")
    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1: {grid_search.best_score_:.4f}")

    # 在独立测试集上评估
    y_pred = grid_search.predict(X_test)
    y_prob = grid_search.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }

    print("\n===== 测试集评估结果 =====")
    for name, value in metrics.items():
        print(f"  {name:<12s}: {value:.4f}")

    return grid_search.best_estimator_, metrics


if __name__ == '__main__':
    DATA_PATH = 'train_data.csv'

    X, y, cat_cols, num_cols = load_and_prepare_data(DATA_PATH)
    best_model, eval_metrics = tune_and_evaluate(X, y, cat_cols, num_cols)
