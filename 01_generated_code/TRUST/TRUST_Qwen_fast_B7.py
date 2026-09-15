import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)


def load_and_prepare_data(filepath: str):
    """读取数据并将TRUST滴度转换为二分类标签"""
    df = pd.read_csv(filepath, encoding='utf-8')

    # 定义特征列与目标列
    feature_cols = [
        'SEX', 'AGE', 'DEPT', 'DIAGNOSIS', 'TPPA',
        'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR'
    ]
    target_col = 'TRUST'

    X = df[feature_cols].copy()
    # TRUST为滴度值(1,2,4,8,...)，转为二分类: >=16 为阳性(1), <16 为阴性(0)
    y = (df[target_col] >= 16).astype(int)

    return X, y


def build_preprocessor():
    """构建包含缺失值填充与编码的预处理管道"""
    categorical_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ord = ['TPPA']
    numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # TPPA为有序分类变量，需手动指定类别顺序（按滴度/反应强度升序）
    # 请根据实际数据中TPPA的取值调整此顺序
    tppa_categories = [['阴性', '弱阳性', '阳性', '强阳性']]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_imputer', SimpleImputer(strategy='median'), numerical_cols),
            ('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_ohe),
            ('cat_ord', OrdinalEncoder(categories=tppa_categories, handle_unknown='use_encoded_value', unknown_value=-1), categorical_ord)
        ],
        remainder='drop'
    )
    return preprocessor


def train_and_evaluate(X, y):
    """训练随机森林模型并输出评估指标"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    # 使用imblearn的Pipeline以支持SMOTE与GridSearchCV联合使用
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

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
        n_jobs=1,          # 不使用多进程，兼容PyCharm调试
        verbose=1,
        refit=True
    )

    print("开始超参数调优...")
    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1: {grid_search.best_score_:.4f}\n")

    # 在测试集上评估
    y_pred = grid_search.predict(X_test)
    y_prob = grid_search.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }

    print("=" * 40)
    print("测试集评估结果:")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"{name:<12}: {value:.4f}")
    print("=" * 40)

    return grid_search


if __name__ == '__main__':
    DATA_PATH = 'train_data.csv'
    X, y = load_and_prepare_data(DATA_PATH)
    model = train_and_evaluate(X, y)
