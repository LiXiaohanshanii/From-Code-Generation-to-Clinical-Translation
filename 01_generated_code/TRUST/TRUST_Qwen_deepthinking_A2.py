import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
# 必须使用 imblearn 的 Pipeline，因为 SMOTE 会改变样本数量，标准 sklearn Pipeline 不支持
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # ==========================================
    # 1. 数据读取与目标变量处理
    # ==========================================
    print("正在读取数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 确保 TRUST 列为数值类型（防止数据中存在非数值字符）
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')

    # 目标变量转换：二分类任务，>=16 为 1，否则为 0
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop('TRUST', axis=1)

    # ==========================================
    # 2. 划分训练集和测试集
    # ==========================================
    # 使用 stratify=y 确保训练集和测试集中的正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 定义特征列与数据预处理
    # ==========================================
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 修正了提示中的拼写错误 DIAGONSIS -> DIAGNOSIS
    cat_ord_cols = ['TPPA']

    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量：独热编码 (One-Hot) 和 序数编码 (Ordinal)
    # sparse_output=False 确保输出为稠密矩阵，以便后续 SMOTE 处理
    cat_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat_onehot', cat_onehot_transformer, cat_onehot_cols),
            ('cat_ord', cat_ord_transformer, cat_ord_cols)
        ]
    )

    # ==========================================
    # 4. 构建 Pipeline (包含 SMOTE 和 随机森林)
    # ==========================================
    smote = SMOTE(random_state=42)
    rf = RandomForestClassifier(random_state=42)

    # 使用 imblearn 的 Pipeline 将预处理、重采样和模型串联
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', rf)
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

    # n_jobs=1 明确禁用多进程，避免在 PyCharm/Windows 环境下出现子进程报错
    # scoring='f1' 针对不平衡数据，使用 F1 分数作为调优的核心指标比 Accuracy 更合理
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,
        verbose=1
    )

    print("开始进行超参数调优与模型训练...")
    grid_search.fit(X_train, y_train)
    print(f"最佳超参数组合: {grid_search.best_params_}")

    # ==========================================
    # 6. 模型评估
    # ==========================================
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "=" * 30)
    print("       模型评估结果")
    print("=" * 30)
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"F1分数 (F1-score) : {f1:.4f}")
    print(f"AUC曲线下面积     : {auc:.4f}")
    print("=" * 30)


if __name__ == '__main__':
    main()
