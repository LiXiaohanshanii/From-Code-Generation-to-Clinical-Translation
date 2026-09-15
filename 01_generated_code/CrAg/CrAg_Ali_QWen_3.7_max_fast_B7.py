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


def load_and_prepare_data(filepath: str):
    """加载数据并将目标列转换为二分类标签"""
    df = pd.read_csv(filepath, encoding='utf-8')

    # 确认特征列与目标列
    feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color', 'Protein',
                    'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS']
    target_col = 'CSF-T'

    X = df[feature_cols].copy()
    y = (df[target_col] >= 20).astype(int)  # ≥20为阳性(1)，<20为阴性(0)

    return X, y


def build_preprocessor():
    """
    构建预处理管道：
    - 连续变量：中位数填充
    - 序数变量(SER-T, Ink staining)：OrdinalEncoder
    - 名义变量(Color, Transparency, SEX, DEPT, DIAGNOSIS)：OneHotEncoder
    """
    numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    ordinal_features = ['SER-T', 'Ink staining']
    nominal_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1), ordinal_features),
            ('nom', OneHotEncoder(handle_unknown='ignore',
                                  sparse_output=False), nominal_features)
        ],
        remainder='drop'
    )
    return preprocessor


def train_and_evaluate(X, y):
    """训练随机森林模型并输出评估指标"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 使用imblearn的Pipeline确保SMOTE仅在训练集上执行，避免数据泄露
    model_pipeline = ImbPipeline([
        ('preprocessor', build_preprocessor()),
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
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,          # 不使用多进程，兼容PyCharm调试
        refit=True,
        verbose=1
    )

    print("=" * 60)
    print("开始超参数搜索...")
    print("=" * 60)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    }

    print("\n" + "=" * 60)
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"交叉验证最佳F1: {grid_search.best_score_:.4f}")
    print("=" * 60)
    print("测试集评估结果:")
    for name, value in metrics.items():
        print(f"  {name:<12s}: {value:.4f}")
    print("=" * 60)

    return best_model, metrics


if __name__ == '__main__':
    DATA_PATH = 'CrAg_train.csv'

    X, y = load_and_prepare_data(DATA_PATH)
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"目标分布: 阴性(<20)={sum(y == 0)}, 阳性(≥20)={sum(y == 1)}")

    model, evaluation_metrics = train_and_evaluate(X, y)