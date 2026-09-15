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

# 使用 imblearn 的 Pipeline 确保 SMOTE 仅在交叉验证的训练折中执行，避免数据泄露
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据集
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构建目标变量（TRUST >= 16 为 1，否则为 0）
    X = df.drop(columns=['TRUST'])
    y = (df['TRUST'] >= 16).astype(int)

    # 3. 划分特征类型
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']

    # 4. 构建数据预处理流水线 (ColumnTransformer)
    # - 连续变量：中位数填充
    # - SEX, DEPT, DIAGNOSIS：独热编码
    # - TPPA：序数编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), continuous_cols),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_cols),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ]
    )

    # 5. 构建完整流水线（预处理 -> SMOTE采样 -> 随机森林分类器）
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 6. 设置网格搜索超参数字典
    # 注意：在 scikit-learn 中，'平衡' 对应的参数值为 'balanced'
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    # 7. 划分训练集与测试集（按目标变量比例分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. 超参数调优（GridSearchCV，单进程执行 n_jobs=1）
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1  # 不使用多进程
    )

    print("开始模型训练与网格搜索调优...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳超参数组合: {grid_search.best_params_}\n")

    # 9. 模型评估
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    # 10. 输出指标结果
    print("=" * 40)
    print("      测试集模型评估指标")
    print("=" * 40)
    print(f"准确率 (Accuracy):   {acc:.4f}")
    print(f"精确率 (Precision):  {prec:.4f}")
    print(f"召回率 (Recall):     {rec:.4f}")
    print(f"F1分数 (F1-score):   {f1:.4f}")
    print(f"ROC-AUC 面积:        {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()