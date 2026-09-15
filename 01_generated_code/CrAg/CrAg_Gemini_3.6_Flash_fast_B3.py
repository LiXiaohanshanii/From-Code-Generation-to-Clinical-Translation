import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据集
    file_path = "CrAg_train.csv"
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"错误：未能找到文件 '{file_path}'，请检查文件路径是否正确。")
        return

    # 2. 构建目标变量（二分类：CSF-T >= 20 为 1，否则为 0）
    target_col = df.columns[-1]  # 最后一列为 CSF-T
    y = (df[target_col] >= 20).astype(int)

    # 特征矩阵
    X = df.drop(columns=[target_col])

    # 3. 定义变量类型
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['SER-T', 'Ink staining']

    # 4. 构建预处理 Pipeline (ColumnTransformer)
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量：One-Hot 编码（设置 handle_unknown='ignore' 以防止测试集中出现未见过的类别导致报错）
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 序数编码：Ordinal Encoder
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ]
    )

    # 5. 划分训练集与测试集 (Stratified 采样保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建带有 SMOTE 的 Imbalance Pipeline
    # 注意：必须使用 imblearn 的 Pipeline，确保 SMOTE 只对训练集生效，不污染评估过程
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 超参数调优设置 (使用 GridSearch)
    # 映射网格参数到 Pipeline 中 classifier 的参数名
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("正在进行网格搜索与超参数调优（以 F1 score 为指标）...")
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,  # 任务指定：不使用多进程
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n最佳超参数组合: {grid_search.best_params_}")
    print(f"交叉验证最佳 F1 均值: {grid_search.best_score_:.4f}")

    # 8. 获取最佳模型并在测试集上进行评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 9. 打印模型评估结果
    print("\n" + "=" * 40)
    print("      测试集模型评估结果 (CSF-T >= 20)")
    print("=" * 40)
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"F1 分数 (F1-score): {f1:.4f}")
    print(f"AUC 面积 (ROC-AUC): {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()