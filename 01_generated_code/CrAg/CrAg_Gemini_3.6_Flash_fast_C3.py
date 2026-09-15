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
    # 1. 读取数据集
    file_path = 'CrAg_train.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 区分特征与目标变量 (目标列 CSF-T 是否 >= 20)
    target_col = 'CSF-T'
    X = df.drop(columns=[target_col])
    y = (df[target_col] >= 20).astype(int)

    # 3. 按要求划分变量类型
    # 连续变量
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    # 独热编码分类变量
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    # 序数编码分类变量
    ordinal_cols = ['SER-T', 'Ink staining']

    # 4. 划分训练集与验证集 (4:1 比例，按类别比例分层抽样)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. 构建预处理流水线 (按要求顺序：数值 -> 独热 -> 序数)
    # 数值变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 独热编码变量
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 序数编码变量
    ordinal_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ],
        remainder='drop'
    )

    # 6. 构建 imblearn Pipeline (防止数据泄露：SMOTE 仅作用于训练 fold)
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 7. 定义网格搜索参数 (不使用多进程 n_jobs=1)
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1
    )

    # 8. 模型训练与超参数调优
    print("正在开始模型训练与参数调优...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳模型参数: {grid_search.best_params_}")

    # 9. 验证集评估
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    # 10. 输出评估指标
    print("\n--- 验证集评估结果 (CSF-T >= 20) ---")
    print(f"准确率 (Accuracy):   {acc:.4f}")
    print(f"召回率 (Recall):     {rec:.4f}")
    print(f"精确率 (Precision):  {prec:.4f}")
    print(f"F1分数 (F1-score):   {f1:.4f}")
    print(f"AUC曲线下面积 (AUC): {auc:.4f}")


if __name__ == '__main__':
    main()