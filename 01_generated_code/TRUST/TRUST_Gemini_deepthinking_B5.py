import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
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
    # 1. 数据加载
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 目标变量生成：TRUST滴度是否 >= 16（1为高滴度/阳性事件，0为低滴度/阴性事件）
    # TRUST 列为数据集最后一列
    target_col = 'TRUST'
    df['TARGET'] = (df[target_col] >= 16).astype(int)

    # 3. 特征分类定义
    # 连续变量（需要中位数填充 missing values）
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    # 分类变量 - 独热编码
    ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    # 分类变量 - 序数编码
    ord_cols = ['TPPA']

    # 提取特征阵与目标向量
    feature_cols = num_cols + ohe_cols + ord_cols
    X = df[feature_cols]
    y = df['TARGET']

    # 4. 构建数据预处理 Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_imputer', SimpleImputer(strategy='median'), num_cols),
            ('ohe_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_cols),
            ('ord_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ord_cols)
        ]
    )

    # 5. 数据集划分（80% 训练集, 20% 测试集，采用分层抽样保持目标比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建 Imbalanced-learn 流水线（预处理 -> SMOTE 过采样 -> 随机森林分类器）
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 7. 设置超参数网格
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    # 8. 超参数调优（GridSearchCV，单线程 n_jobs=1，以 F1-score 为优化目标）
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,
        verbose=1
    )

    print("开始模型训练与参数调优...")
    grid_search.fit(X_train, y_train)

    # 9. 获取最佳模型并在测试集上评估
    best_model = grid_search.best_estimator_

    # 预测分类标签与概率值
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 10. 输出结果
    print("\n" + "=" * 40)
    print("【最佳超参数组合】")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")

    print("\n【测试集模型评估结果】")
    print(f"  准确率 (Accuracy):        {acc:.4f}")
    print(f"  精确率 (Precision):       {prec:.4f}")
    print(f"  召回率 (Recall):          {rec:.4f}")
    print(f"  F1 分数 (F1-score):       {f1:.4f}")
    print(f"  ROC-AUC (曲线下面积):     {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()