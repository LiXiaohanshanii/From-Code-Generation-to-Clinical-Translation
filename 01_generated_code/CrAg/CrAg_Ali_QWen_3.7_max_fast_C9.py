import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
import warnings

# 忽略不必要的警告信息，保持输出整洁
warnings.filterwarnings('ignore')


def main():
    # ==========================
    # 1. 数据加载与目标变量构建
    # ==========================
    print(">>> 正在加载数据...")
    df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

    # 特征列定义
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    cat_onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    cat_ordinal_cols = ['SER-T', 'Ink staining']

    feature_cols = num_cols + cat_onehot_cols + cat_ordinal_cols

    # 构建二分类目标：CSF-T >= 20 为阳性(1)，否则为阴性(0)
    y = (df['CSF-T'] >= 20).astype(int)
    X = df[feature_cols]

    print(f">>> 数据集形状: {X.shape}, 阳性样本比例: {y.mean():.4f}")

    # ==========================
    # 2. 划分训练集与验证集 (4:1)
    # ==========================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f">>> 训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

    # ==========================
    # 3. 构建防泄漏预处理管道
    # ==========================
    # 注意：预处理顺序严格按照 数值 -> 独热 -> 序数

    # 3.1 数值型变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 3.2 独热编码变量
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 3.3 序数编码变量 (SER-T, Ink staining)
    # 由于是序数变量，需根据实际临床意义排序，此处假设按字符串自然排序或自定义
    # 若需自定义顺序，请修改categories参数，例如: categories=[['低','中','高'], [...]]
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value',
                                         unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, cat_onehot_cols),
            ('ordinal', ordinal_transformer, cat_ordinal_cols)
        ],
        remainder='drop'
    )

    # 3.4 使用 imblearn Pipeline 将 SMOTE 嵌入管道内部
    # 关键点：SMOTE 仅在 fit 阶段对训练数据生效，transform/predict 时自动跳过
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================
    # 4. 超参数调优 (GridSearchCV)
    # ==========================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print(">>> 开始超参数调优 (以F1为评分指标)...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,  # 5折交叉验证，SMOTE在每折内部执行
        n_jobs=1,  # 不使用多进程
        verbose=1
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
    print("       验证集模型评估结果")
    print("=" * 40)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()