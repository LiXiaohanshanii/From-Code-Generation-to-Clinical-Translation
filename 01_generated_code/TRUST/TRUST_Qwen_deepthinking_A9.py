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
    # 1. 加载数据
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 确保目标列为数值类型，防止因格式问题导致比较出错
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')

    # ==========================================
    # 2. 定义特征与目标变量
    # ==========================================
    # 目标变量二值化：>=16 为 1，否则为 0
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # 划分训练集和测试集 (80% 训练, 20% 测试)，使用分层抽样保持类别比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 定义特征列分组
    # ==========================================
    # 修正了题目描述中 DIAGONSIS 的拼写错误为 DIAGNOSIS
    categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ordinal_cols = ['TPPA']
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # ==========================================
    # 4. 构建数据预处理器 (ColumnTransformer)
    # ==========================================
    # 连续变量：中位数填充
    continuous_transformer = SimpleImputer(strategy='median')

    # 独热编码 (One-Hot Encoding) - 必须设置 sparse_output=False 以便 SMOTE 处理
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 序数编码 (Ordinal Encoding)
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', continuous_transformer, continuous_cols),
            ('onehot', onehot_transformer, categorical_onehot_cols),
            ('ordinal', ordinal_transformer, categorical_ordinal_cols)
        ]
    )

    # ==========================================
    # 5. 构建包含 SMOTE 的 Pipeline
    # ==========================================
    # 使用 imblearn 的 Pipeline，确保 SMOTE 在数据预处理（编码和填充）之后执行
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 6. 超参数调优 (GridSearchCV)
    # ==========================================
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10],
        'clf__min_samples_split': [2],
        'clf__min_samples_leaf': [1],
        'clf__class_weight': ['balanced']
    }

    print("正在进行超参数调优（不使用多进程），请稍候...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',  # 针对不平衡数据，使用 AUC 作为调优的评价标准
        cv=5,
        n_jobs=1,  # 明确指定 n_jobs=1 以禁用多进程
        verbose=1
    )

    # 在训练集上拟合模型
    grid_search.fit(X_train, y_train)

    # ==========================================
    # 7. 模型评估
    # ==========================================
    best_model = grid_search.best_estimator_

    # 测试集预测
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # ==========================================
    # 8. 输出结果
    # ==========================================
    print("\n" + "=" * 40)
    print("模型训练与评估完成！")
    print("=" * 40)
    print(f"最佳超参数组合: {grid_search.best_params_}")
    print("-" * 40)
    print(f"准确率 (Accuracy)  : {accuracy:.4f}")
    print(f"召回率 (Recall)    : {recall:.4f}")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积 (AUC): {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
