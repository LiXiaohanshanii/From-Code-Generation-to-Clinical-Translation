import pandas as pd
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
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 加载数据集
    file_path = 'CrAg_train.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构造二分类目标变量 (CSF-T >= 20 为 1，其余为 0)
    target_col = 'CSF-T'
    X = df.drop(columns=[target_col])
    y = (df[target_col] >= 20).astype(int)

    # 3. 按规定顺序与类型定义特征列
    # 顺序要求：连续变量(数值) -> 独热编码变量 -> 序数编码变量
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['SER-T', 'Ink staining']

    # 4. 构建数据预处理列转换器 (ColumnTransformer)
    # 按要求顺序添加转换步骤：数值中位数填充 -> 独热编码 -> 序数编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ]
    )

    # 5. 划分训练集与验证集 (比例 4:1，分层抽样保持类分布)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 6. 构建 imblearn 流水线 (包含预处理、SMOTE过采样、随机森林分类器)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 超参数网格配置
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 8. 使用 GridSearchCV 进行超参数调优 (不使用多进程 n_jobs=1，以 f1 指标评分)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    # 9. 模型拟合
    grid_search.fit(X_train, y_train)

    # 10. 获取最佳模型并在验证集上评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]

    # 11. 计算评估指标
    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    # 12. 输出最佳参数与模型性能结果
    print("=== 超参数调优最佳参数 ===")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")

    print("\n=== 验证集模型评估指标 ===")
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"F1 分数 (F1-score) : {f1:.4f}")
    print(f"AUC 面积 (ROC AUC) : {auc:.4f}")


if __name__ == '__main__':
    main()