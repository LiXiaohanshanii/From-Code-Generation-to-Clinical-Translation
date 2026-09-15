import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
# 关键导入：使用imblearn的Pipeline以支持SMOTE嵌入交叉验证
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def main():
    # ==========================
    # 1. 数据加载与目标变量转换
    # ==========================
    print(">>> 正在加载数据...")
    df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

    # 定义特征列
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['SER-T', 'Ink staining']

    feature_cols = num_cols + onehot_cols + ordinal_cols

    X = df[feature_cols]
    y_raw = df['CSF-T'].astype(float)

    # 二分类转换: >=20 为阳性(1), <20 为阴性(0)
    y = (y_raw >= 20).astype(int)

    print(f">>> 数据集形状: {X.shape}")
    print(f">>> 目标变量分布:\n{y.value_counts(normalize=True)}")

    # ==========================
    # 2. 划分训练集与验证集 (4:1)
    # ==========================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f">>> 训练集样本数: {len(X_train)}, 验证集样本数: {len(X_val)}")

    # ==========================
    # 3. 构建防泄露预处理管道
    # ==========================
    # 预处理顺序: 数值缺失值填充 -> 独热编码 -> 序数编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    # 使用 imblearn.pipeline.Pipeline 确保 SMOTE 只在 CV 训练折内执行
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================
    # 4. 超参数调优 (基于F1, 单进程)
    # ==========================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print(">>> 开始网格搜索超参数调优 (评分指标: F1)...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,  # 不使用多进程，兼容PyCharm调试
        verbose=1,
        refit=True  # 用最佳参数在全部训练集上重新拟合
    )

    grid_search.fit(X_train, y_train)

    print(f">>> 最佳参数: {grid_search.best_params_}")
    print(f">>> 最佳交叉验证 F1: {grid_search.best_score_:.4f}")

    # ==========================
    # 5. 验证集评估
    # ==========================
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print("\n" + "=" * 40)
    print(">>> 验证集模型评估结果")
    print("=" * 40)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()