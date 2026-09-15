import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据
    # 请确保文件 CrAg_train.csv 与代码位于同一目录下，或填写绝对路径
    file_path = 'CrAg_train.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构造二分类目标变量 (CSF-T >= 20 为 1，否则为 0)
    # 目标列为最后一列 CSF-T
    target_col = 'CSF-T'
    y = (df[target_col] >= 20).astype(int)
    X = df.drop(columns=[target_col])

    # 3. 特征分组定义
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

    # 序数编码变量定义
    # 注意：这里需要根据实际数据的类别顺序调整 categories 列表
    # 如果实际数据中的字符串格式不同，请调整以下列表中对应的值
    ordinal_cols = ['SER-T', 'Ink staining']

    # 独热编码变量定义
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    # 4. 构建数据预处理流水线 (ColumnTransformer)
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 序数编码：按显式顺序处理
    # 如果数据集中的 SER-T 或 Ink staining 包含具体数值/文本顺序，可在此处指定类别列表
    # 此处设为 'auto'，自动按字母/数值顺序编码（亦可传自定义 categories 列表）
    ordinal_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    # 独热编码：处理未在训练集见过的分类特征
    onehot_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('ord', ordinal_transformer, ordinal_cols),
            ('cat', onehot_transformer, onehot_cols)
        ]
    )

    # 5. 划分训练集与测试集 (80% 训练, 20% 测试，采用分层抽样保持正负样本比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建包含 SMOTE 和随机森林的 Pipeline (使用 imblearn pipeline 避免数据渗漏)
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 定义超参数网格 (完全对应指定参数，禁用多进程 n_jobs=1)
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
        n_jobs=1,  # 禁用多进程
        verbose=1
    )

    # 8. 模型训练与超参数寻优
    print("开始模型训练与网格搜索...")
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print("\n最佳模型参数：", grid_search.best_params_)

    # 9. 模型预测与评估
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正例概率值计算 AUC

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 10. 打印评估结果
    print("\n================== 模型评估指标 (Test Set) ==================")
    print(f"准确率 (Accuracy):   {accuracy:.4f}")
    print(f"精确率 (Precision):  {precision:.4f}")
    print(f"召回率 (Recall):     {recall:.4f}")
    print(f"F1 分数 (F1-Score):  {f1:.4f}")
    print(f"ROC-AUC 面积:        {roc_auc:.4f}")
    print("=============================================================")


if __name__ == '__main__':
    main()