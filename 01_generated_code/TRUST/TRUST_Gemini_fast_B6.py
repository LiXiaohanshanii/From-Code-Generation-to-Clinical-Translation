import os
import pandas as pd
import numpy as np

# 导入 sklearn 模块
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

# 注意：为了避免交叉验证中的数据泄露，SMOTE 必须通过 imblearn.pipeline 组装
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def build_and_evaluate_rf():
    # ---------------------------------------------------------
    # 1. 数据读取与列名校验
    # ---------------------------------------------------------
    data_path = 'train_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据集文件: {data_path}，请确认文件路径是否正确。")

    df = pd.read_csv(data_path, encoding='utf-8')

    # 自动修正拼写兼容（处理 DIAGONSIS / DIAGNOSIS 错别字）
    if 'DIAGONSIS' in df.columns and 'DIAGNOSIS' not in df.columns:
        df.rename(columns={'DIAGONSIS': 'DIAGNOSIS'}, inplace=True)

    # ---------------------------------------------------------
    # 2. 目标变量构建与特征分离
    # ---------------------------------------------------------
    # 目标列为最后一列 TRUST，转换为二分类变量：TRUST >= 16 为 1，否则为 0
    target_col = 'TRUST'

    X = df.drop(columns=[target_col])
    y = (df[target_col] >= 16).astype(int)

    # 明确变量类型分组
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']

    # ---------------------------------------------------------
    # 3. 预处理流水线构建 (ColumnTransformer)
    # ---------------------------------------------------------
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 无序分类变量：独热编码
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 有序分类变量：序数编码
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ]
    )

    # ---------------------------------------------------------
    # 4. 构建包含 SMOTE 和随机森林的集成流水线
    # ---------------------------------------------------------
    # 使用 imblearn 的 Pipeline 可以在交叉验证中仅对训练折进行 SMOTE 采样，防止数据泄露
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ---------------------------------------------------------
    # 5. 超参数搜索空间定义
    # ---------------------------------------------------------
    # 说明：题目要求 class_weight 为 '平衡'，在 sklearn 中对应参数值为 'balanced'
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # ---------------------------------------------------------
    # 6. 数据集划分与网格搜索拟合
    # ---------------------------------------------------------
    # 按分层抽样划分训练集与测试集（80% 训练，20% 测试）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 网格搜索超参数（指定单线程执行：n_jobs=1）
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=1
    )

    print("开始模型拟合与网格搜索调参...")
    grid_search.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 7. 模型评估
    # ---------------------------------------------------------
    best_model = grid_search.best_estimator_

    # 预测类别和预测概率
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 计算各个评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # ---------------------------------------------------------
    # 8. 输出评估结果
    # ---------------------------------------------------------
    print("\n================ 模型搜索结果 ================")
    print("最优超参数配置:", grid_search.best_params_)
    print("\n================ 模型测试集评估指标 ================")
    print(f"准确率 (Accuracy)   : {acc:.4f}")
    print(f"召回率 (Recall)     : {rec:.4f}")
    print(f"精确率 (Precision)  : {prec:.4f}")
    print(f"F1 分数 (F1-score)   : {f1:.4f}")
    print(f"AUC 曲线下面积 (AUC): {auc:.4f}")
    print("==================================================")


if __name__ == '__main__':
    build_and_evaluate_rf()