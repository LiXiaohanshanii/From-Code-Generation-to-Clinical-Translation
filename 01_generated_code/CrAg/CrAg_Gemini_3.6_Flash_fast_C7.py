import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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
    # 1. 读取数据
    df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

    # 2. 构建目标列：CSF-T >= 20 为正例 (1)，< 20 为负例 (0)
    y = (df['CSF-T'] >= 20).astype(int)
    X = df.drop(columns=['CSF-T'])

    # 3. 按类型划分特征
    continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['SER-T', 'Ink staining']

    # 4. 构建数据预处理 ColumnTransformer（严格顺序：数值中位数填充 -> 独热编码 -> 序数编码）
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), continuous_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ],
        remainder='drop'
    )

    # 5. 划分训练集与验证集（4:1 比例，按类别分层抽样）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 使用 imblearn 的 Pipeline 嵌入 SMOTE，防止交叉验证过程中的数据泄露
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 超参数网格设置（指定单进程 n_jobs=1）
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
        n_jobs=1
    )

    # 8. 执行模型训练与调优
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("最佳超参数组合：", grid_search.best_params_)

    # 9. 在独立验证集上评估模型
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    # 10. 输出指标
    print("\n================ 验证集评估指标 ================")
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}")
    print(f"精确率 (Precision) : {prec:.4f}")
    print(f"F1 分数 (F1-score) : {f1:.4f}")
    print(f"AUC 面积 (ROC-AUC) : {auc:.4f}")


if __name__ == '__main__':
    main()