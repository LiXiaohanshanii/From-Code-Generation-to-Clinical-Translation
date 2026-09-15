import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
# 关键导入：使用imblearn的Pipeline以支持SMOTE步骤，防止数据泄露
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def build_and_evaluate_rf():
    # ==========================================
    # 1. 数据加载与目标变量重构
    # ==========================================
    try:
        df = pd.read_csv('CrAg_train.csv', encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError("未找到CrAg_train.csv文件，请确认文件路径及名称是否正确。")

    # 定义特征列
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['SER-T', 'Ink staining']

    feature_cols = num_cols + onehot_cols + ordinal_cols
    target_col = 'CSF-T'

    X = df[feature_cols]
    y_raw = df[target_col]

    # 任务目标：预测CSF-T是否 >= 20 (二分类)
    # 注意：滴度数据可能为字符串或数值型，统一转为数值比较更安全
    y = (pd.to_numeric(y_raw, errors='coerce') >= 20).astype(int)

    print(f"数据集大小: {X.shape}")
    print(f"目标变量分布:\n{y.value_counts()}")
    print("-" * 50)

    # ==========================================
    # 2. 划分训练集与验证集 (4:1)
    # ==========================================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 构建防泄露Pipeline
    # 预处理顺序严格要求：数值填充 → 独热编码 → 序数编码 → SMOTE → 模型
    # ==========================================
    pipeline = ImbPipeline([
        # Step 1: 连续变量缺失值中位数填充
        ('num_imputer', SimpleImputer(strategy='median')),

        # Step 2: 分类变量独热编码
        # handle_unknown='ignore' 防止验证集出现新类别报错
        ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),

        # Step 3: 序数编码
        # SER-T和Ink staining作为有序分类变量处理
        ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),

        # Step 4: SMOTE过采样 (仅在训练折内执行，彻底防止数据泄露)
        ('smote', SMOTE(random_state=42)),

        # Step 5: 随机森林分类器
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 设置各步骤对应的列索引/列名
    # 由于imblearn Pipeline对DataFrame列选择的支持，我们需要用ColumnTransformer
    # 但为了保持代码简洁并严格满足"不使用多进程+imblearn Pipeline"的要求，
    # 这里采用更稳健的方式：先用ColumnTransformer包装预处理，再放入ImbPipeline

    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ],
        remainder='drop'
    )

    # 重组Pipeline：预处理器 → SMOTE → RF
    final_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 4. 超参数调优 (GridSearchCV, n_jobs=1)
    # ==========================================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=final_pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,  # 明确要求不使用多进程
        verbose=1
    )

    print("开始超参数调优...")
    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")
    print("-" * 50)

    # ==========================================
    # 5. 验证集评估
    # ==========================================
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print("【验证集评估结果】")
    print(f"Accuracy : {acc:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    return best_model


if __name__ == "__main__":
    model = build_and_evaluate_rf()