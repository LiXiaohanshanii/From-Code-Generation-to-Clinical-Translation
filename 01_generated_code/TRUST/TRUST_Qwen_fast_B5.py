import warnings
import numpy as np
import pandas as pd
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

# 忽略无关警告信息
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath: str):
    """
    加载数据并将目标列TRUST转换为二分类标签(>=16为1, <16为0)
    """
    df = pd.read_csv(filepath, encoding='utf-8')

    # 定义特征列
    categorical_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']
    numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    feature_cols = categorical_cols + ordinal_cols + numerical_cols
    X = df[feature_cols].copy()

    # 目标列二值化: TRUST >= 16 -> 1, 否则 -> 0
    y = (df['TRUST'] >= 16).astype(int)

    return X, y, categorical_cols, ordinal_cols, numerical_cols


def build_pipeline(categorical_cols, ordinal_cols, numerical_cols):
    """
    构建包含预处理和随机森林的imblearn Pipeline
    注意: SMOTE必须在Pipeline内部以防止测试集数据泄露
    """
    # 1. 连续变量: 中位数填充
    num_transformer = Pipeline(steps=[
        ('median_imputer', SimpleImputer(strategy='median'))
    ])

    # 2. 分类变量: 独热编码 (handle_unknown='ignore'防止测试集出现新类别报错)
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. 序数变量: TPPA序数编码
    # 假设TPPA的有序类别为阴性到强阳性，请根据实际数据调整categories
    ord_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(
            categories=[['Negative', 'Weak Positive', 'Positive', 'Strong Positive']],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols),
            ('ord', ord_transformer, ordinal_cols)
        ],
        remainder='drop'
    )

    # 使用imblearn的Pipeline以支持SMOTE步骤
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return pipeline


def evaluate_model(y_true, y_pred, y_prob):
    """计算并打印所有指定评估指标"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob)
    }

    print("\n" + "=" * 50)
    print("       模型评估结果")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print("=" * 50)

    return metrics


def main():
    # ===================== 配置区 =====================
    DATA_PATH = 'train_data.csv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # 超参数网格 (不使用多进程, n_jobs=1)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }
    # ==================================================

    # 1. 加载与准备数据
    print("[1/5] 加载数据并进行目标变量二值化...")
    X, y, cat_cols, ord_cols, num_cols = load_and_prepare_data(DATA_PATH)
    print(f"      数据集形状: {X.shape}, 正样本比例: {y.mean():.4f}")

    # 2. 划分训练集与测试集 (SMOTE仅作用于训练集)
    print("[2/5] 划分训练集与测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. 构建Pipeline
    print("[3/5] 构建预处理+SMOTE+随机森林Pipeline...")
    pipe = build_pipeline(cat_cols, ord_cols, num_cols)

    # 4. 超参数调优 (GridSearchCV, 单进程)
    print("[4/5] 开始网格搜索超参数调优 (n_jobs=1)...")
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='f1',          # 不平衡数据以F1为优化目标
        cv=5,
        n_jobs=1,              # 明确要求不使用多进程
        verbose=1,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    print(f"\n      最佳参数: {grid_search.best_params_}")
    print(f"      最佳交叉验证F1: {grid_search.best_score_:.4f}")

    # 5. 测试集评估
    print("[5/5] 在测试集上评估最优模型...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_pred, y_prob)


if __name__ == '__main__':
    main()
