import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
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
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据
    # 注意：如果文件不在当前工作目录下，请更换为实际路径
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 定义特征和目标变量
    # 根据要求构建二分类目标列：TRUST >= 16 为 1，否则为 0
    target_col = df.columns[-1]  # 数据集最后一列 (TRUST)
    y = (df[target_col] >= 16).astype(int)
    X = df.drop(columns=[target_col])

    # 列分类
    categorical_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ord = ['TPPA']
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 3. 特征预处理 Pipeline 构建
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 无序分类变量：独热编码
    # handle_unknown='ignore' 可以防止测试数据出现未知类别时报错
    ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 有序分类变量：序数编码
    ord_transformer = OrdinalEncoder()

    # 组合 ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, continuous_cols),
            ('ohe', ohe_transformer, categorical_ohe),
            ('ord', ord_transformer, categorical_ord)
        ]
    )

    # 4. 执行数据预处理与特征转换
    X_processed = preprocessor.fit_transform(X)

    # 5. 处理类别不平衡问题 (SMOTE)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)

    # 6. 超参数调优与随机森林模型构建
    # 注意：参数字典中的 'balanced' 对应 sklearn 中的 'balanced' 字符串
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']  # 映射中文‘平衡’为 sklearn 标准参数 'balanced'
    }

    # 基础模型声明
    rf_base = RandomForestClassifier(random_state=42)

    # 网格搜索（设置 n_jobs=1 禁用多进程）
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1  # 不使用多进程
    )

    # 拟合模型
    grid_search.fit(X_resampled, y_resampled)
    best_model = grid_search.best_estimator_

    # 7. 模型预测与评估（基于训练/过采样后数据集的预测评估）
    y_pred = best_model.predict(X_resampled)
    y_pred_proba = best_model.predict_proba(X_resampled)[:, 1]

    accuracy = accuracy_score(y_resampled, y_pred)
    recall = recall_score(y_resampled, y_pred)
    precision = precision_score(y_resampled, y_pred)
    f1 = f1_score(y_resampled, y_pred)
    auc = roc_auc_score(y_resampled, y_pred_proba)

    # 8. 输出最佳参数及评估结果
    print("=" * 40)
    print("最佳模型参数配置：")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")

    print("\n模型评估结果：")
    print(f"  准确率 (Accuracy) : {accuracy:.4f}")
    print(f"  召回率 (Recall)   : {recall:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  F1 分数 (F1-score): {f1:.4f}")
    print(f"  ROC AUC 面积     : {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()