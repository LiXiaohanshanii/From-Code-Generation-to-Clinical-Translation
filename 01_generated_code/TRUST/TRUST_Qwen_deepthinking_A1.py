import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # ==========================================
    # 1. 加载与初步处理数据
    # ==========================================
    # 加载数据集
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 确保目标列TRUST为数值类型，并丢弃目标列缺失的行
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')
    df = df.dropna(subset=['TRUST'])

    # 目标变量二值化：>=16 为 1，否则为 0
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # ==========================================
    # 2. 划分训练集与测试集
    # ==========================================
    # 使用分层抽样(stratify)确保训练集和测试集中正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 定义特征列与预处理器
    # ==========================================
    # 注：任务描述中的 'DIAGONSIS' 修正为正确的拼写 'DIAGNOSIS'
    cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ordinal_cols = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 连续变量：中位数填充缺失值
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量（独热编码）：无缺失值，处理未知类别
    cat_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量（序数编码）：无缺失值，处理未知类别
    cat_ordinal_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat_onehot', cat_onehot_transformer, cat_onehot_cols),
            ('cat_ordinal', cat_ordinal_transformer, cat_ordinal_cols)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    # ==========================================
    # 4. 构建包含 SMOTE 的 Pipeline
    # ==========================================
    # 使用 imblearn 的 Pipeline 确保 SMOTE 仅在交叉验证的训练折叠中应用，防止数据泄露
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # n_jobs=1 确保不使用多进程
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',  # 针对不平衡二分类，使用 AUC 作为调优评估指标
        cv=5,
        n_jobs=1,
        verbose=1
    )

    print("开始进行超参数调优与模型训练...")
    grid_search.fit(X_train, y_train)
    print("模型训练完成！\n")

    # ==========================================
    # 6. 模型评估
    # ==========================================
    best_model = grid_search.best_estimator_

    # 预测类别与预测概率
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 打印结果
    print("=" * 40)
    print("最佳超参数组合:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param.replace('classifier__', '')}: {value}")
    print("=" * 40)
    print("测试集评估指标:")
    print(f"  准确率 (Accuracy)  : {acc:.4f}")
    print(f"  召回率 (Recall)    : {rec:.4f}")
    print(f"  精确率 (Precision) : {prec:.4f}")
    print(f"  F1分数 (F1-score)  : {f1:.4f}")
    print(f"  AUC曲线下面积      : {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
