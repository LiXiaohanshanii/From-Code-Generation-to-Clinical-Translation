import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def main():
    # 1. 加载数据集
    file_path = 'CrAg_train.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构建目标列二分类变量 (CSF-T >= 20 为 1，否则为 0)
    # 目标列 CSF-T 位于最后一列
    target_col = 'CSF-T'
    df['target'] = (df[target_col] >= 20).astype(int)

    # 划分特征矩阵 X 与目标向量 y
    X = df.drop(columns=[target_col, 'target'])
    y = df['target']

    # 3. 按要求划分变量类型
    # 连续变量
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    # 独热编码分类变量
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    # 序数编码分类变量
    ordinal_cols = ['SER-T', 'Ink staining']

    # 4. 划分训练集与验证集 (比例 4:1，分层抽样保持类别比例)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 5. 构建预处理流水线 (严格按照：数值中位数填充 -> 独热编码 -> 序数编码 顺序)
    num_transformer = SimpleImputer(strategy='median')
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ],
        remainder='drop'
    )

    # 6. 使用 imblearn.pipeline 组装包含 SMOTE 的完整模型流水线（防止交叉验证中的数据泄露）
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 超参数网格配置 (注意参数名前缀对应 pipeline 中的步骤名称)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 8. 网格搜索调优 (不使用多进程 n_jobs=1，以 F1 分数为评分指标)
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1
    )

    # 运行模型训练与超参数优化
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 9. 在验证集上评估模型
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_proba)

    # 10. 打印评估结果
    print("=" * 40)
    print(" 最佳模型超参数参数:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    print("=" * 40)
    print(" 验证集评估指标结果:")
    print(f"  - 准确率 (Accuracy) : {accuracy:.4f}")
    print(f"  - 召回率 (Recall)   : {recall:.4f}")
    print(f"  - 精确率 (Precision): {precision:.4f}")
    print(f"  - F1 分数 (F1-score) : {f1:.4f}")
    print(f"  - ROC-AUC 曲线面积  : {auc:.4f}")
    print("=" * 40)

if __name__ == '__main__':
    main()