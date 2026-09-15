import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 加载数据集
    file_path = 'CrAg_train.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 目标变量处理：预测 CSF-T 是否 ≥ 20（二分类任务）
    # 确保转换为数值类型，将 >= 20 映射为 1，< 20 映射为 0
    target_col = df.columns[-1]  # 最后一列为 CSF-T
    df['target'] = (pd.to_numeric(df[target_col], errors='coerce') >= 20).astype(int)

    # 3. 区分特征与目标变量
    feature_cols = [
        'SER-T', 'CL', 'GLU', 'Ink staining', 'Color',
        'Protein', 'Transparency', 'RBC', 'SEX', 'AGE', 'DEPT', 'DIAGNOSIS'
    ]
    X = df[feature_cols]
    y = df['target']

    # 4. 按 4:1 划分训练集和验证集（使用分层抽样保持类比例）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. 定义变量类型
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['SER-T', 'Ink staining']

    # 6. 构建数据预处理 ColumnTransformer（严格按要求顺序：数值 → 独热 → 序数）
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ],
        remainder='drop'
    )

    # 7. 构建 imblearn Pipeline（防止数据泄露：SMOTE 仅在交叉验证训练 fold 中执行）
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 8. 网格搜索超参数配置
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 5折交叉验证，评估指标设置为 f1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=1,  # 不使用多进程
        verbose=1
    )

    # 9. 模型训练与调优
    print("开始模型训练与超参数调优...")
    grid_search.fit(X_train, y_train)

    print("\n最佳模型参数：")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")

    best_model = grid_search.best_estimator_

    # 10. 验证集推理与评估
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    # 11. 打印评估结果
    print("\n" + "="*40)
    print("验证集模型评估结果（Validation Performance）")
    print("="*40)
    print(f"准确率 (Accuracy)   : {accuracy:.4f}")
    print(f"召回率 (Recall)     : {recall:.4f}")
    print(f"精确率 (Precision)  : {precision:.4f}")
    print(f"F1 分数 (F1-score)  : {f1:.4f}")
    print(f"ROC-AUC (Area Under ROC) : {auc:.4f}")
    print("="*40)


if __name__ == '__main__':
    main()