import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
    file_path = 'train_data.csv'
    print(f"正在读取数据文件: {file_path} ...")
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构建目标特征（判断TRUST滴度是否 >= 16）
    # 目标列TRUST位于最后一列
    target_col = df.columns[-1]
    y = (df[target_col] >= 16).astype(int)
    X = df.drop(columns=[target_col])

    # 定义特征分类
    onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['TPPA']
    continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 3. 划分数据集（采用分层抽样保持类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 构建数据预处理流水线 (ColumnTransformer)
    # 连续变量：中位数填充
    continuous_transformer = SimpleImputer(strategy='median')

    # One-Hot编码分类变量（handle_unknown='ignore'防止测试集出现新类别时报错）
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 序数编码分类变量
    ordinal_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    # 组合特征转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_transformer, continuous_features),
            ('cat_onehot', onehot_transformer, onehot_features),
            ('cat_ordinal', ordinal_transformer, ordinal_features)
        ]
    )

    # 5. 构建包含SMOTE和随机森林的全流程Pipeline
    # 使用 imblearn 的 Pipeline 确保 SMOTE 仅作用于训练集，防止数据泄漏
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 6. 配置超参数网格与网格搜索 (GridSearchCV)
    # 注意：sklearn中 class_weight='balanced' 即对应“平衡”模式
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    print("开始模型超参数调优...")
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,  # 严格按要求不使用多进程
        verbose=1
    )

    # 模型训练
    grid_search.fit(X_train, y_train)

    print("\n模型训练完成！最优超参数参数组合：")
    for param, val in grid_search.best_params_.items():
        print(f"  {param}: {val}")

    # 7. 在测试集上进行预测与评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 8. 输出评估结果
    print("\n================ 模型测试集评估报告 ================")
    print(f"准确率 (Accuracy)         : {acc:.4f}")
    print(f"召回率 (Recall)           : {rec:.4f}")
    print(f"精确率 (Precision)        : {prec:.4f}")
    print(f"F1 分数 (F1-score)        : {f1:.4f}")
    print(f"ROC曲线下面积 (ROC-AUC)   : {auc:.4f}")
    print("====================================================")


if __name__ == '__main__':
    main()