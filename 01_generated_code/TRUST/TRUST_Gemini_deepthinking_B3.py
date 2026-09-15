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

# 使用 imblearn 提供的 Pipeline，确保 SMOTE 仅在交叉验证的训练折中执行
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def main():
    # 1. 读取数据
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 构建预测目标 TRUST >= 16 (二分类任务)
    # 1 表示 TRUST 滴度 >= 16，0 表示 < 16
    X = df.drop(columns=['TRUST'])
    y = (df['TRUST'] >= 16).astype(int)

    # 3. 定义不同类型特征列名
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']

    # 4. 构建数据预处理转换器 (ColumnTransformer)
    # - 连续变量：中位数填充
    # - SEX, DEPT, DIAGNOSIS：独热编码 (One-Hot)
    # - TPPA：序数编码 (Ordinal)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ]
    )

    # 5. 构建端到端机器学习流水线
    # 流程：预处理 -> SMOTE过采样处理不平衡 -> 随机森林分类器
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 6. 划分训练集与测试集 (80% 训练, 20% 测试，分层抽样保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. 设定网格搜索超参数
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    # 8. 超参数调优 (GridSearchCV)
    # 按照要求设置 n_jobs=1，不使用多进程
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1
    )

    print("开始模型训练与超参数调优...")
    grid_search.fit(X_train, y_train)

    # 9. 获取最佳模型并对测试集进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 10. 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 11. 打印评估结果
    print("\n" + "=" * 40)
    print("      最佳超参数配置      ")
    print("=" * 40)
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")

    print("\n" + "=" * 40)
    print("      模型评估结果 (测试集)      ")
    print("=" * 40)
    print(f"准确率 (Accuracy)   : {acc:.4f}")
    print(f"召回率 (Recall)     : {rec:.4f}")
    print(f"精确率 (Precision)  : {prec:.4f}")
    print(f"F1 分数 (F1-score)  : {f1:.4f}")
    print(f"ROC-AUC 面积 (AUC)  : {auc:.4f}")
    print("=" * 40)

if __name__ == '__main__':
    main()