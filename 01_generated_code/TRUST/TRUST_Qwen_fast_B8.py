import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')


def main():
    # ===================== 1. 数据加载与目标变量构建 =====================
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 定义特征列
    cat_cols = ['SEX', 'DEPT', 'DIAGNOSIS', 'TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    feature_cols = cat_cols + num_cols

    X = df[feature_cols].copy()
    # 二分类任务：TRUST >= 16 为正样本(1)，否则为负样本(0)
    y = (df['TRUST'] >= 16).astype(int)

    # ===================== 2. 划分训练集与测试集 =====================
    # 注意：必须在SMOTE之前划分，防止合成样本泄露到测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ===================== 3. 构建预处理Pipeline =====================
    # 分类变量处理：前3个独热编码，第4个(TPPA)序数编码
    # 由于ColumnTransformer对同一类型列需统一处理，这里拆分为两个transformer
    ohe_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ord_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', ohe_transformer, ['SEX', 'DEPT', 'DIAGNOSIS']),
            ('ord', ord_transformer, ['TPPA']),
            ('num', num_transformer, num_cols)
        ],
        remainder='drop'
    )

    # ===================== 4. 构建含SMOTE的完整Pipeline =====================
    # 使用imblearn的Pipeline确保SMOTE仅在训练集的交叉验证折叠内执行
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ===================== 5. 超参数调优（不使用多进程） =====================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='f1',          # 不平衡数据以F1为优化目标
        cv=5,
        n_jobs=1,              # 明确要求不使用多进程
        verbose=1,
        refit=True
    )

    print("开始超参数调优...")
    grid_search.fit(X_train, y_train)
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}\n")

    # ===================== 6. 模型评估 =====================
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("=" * 50)
    print("       测试集模型评估结果")
    print("=" * 50)
    print(f"准确率 (Accuracy):  {acc:.4f}")
    print(f"召回率 (Recall):    {rec:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"F1分数 (F1-score):  {f1:.4f}")
    print(f"AUC曲线下面积:      {auc:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
