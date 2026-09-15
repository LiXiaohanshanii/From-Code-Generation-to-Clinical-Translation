import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # 1. 加载数据集
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构建二分类目标变量：TRUST >= 16 记为 1，否则记为 0
    # 假设 TRUST 最后一列为数值类型或可直接用于比对
    X = df.drop(columns=['TRUST'])
    y = (df['TRUST'] >= 16).astype(int)

    # 3. 划分特征类型
    # 独热编码特征
    cat_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    # 序数编码特征
    cat_ord_cols = ['TPPA']
    # 连续变量（使用中位数填充）
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 构建数据预处理 Pipeline
    # 连续变量中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量编码
    ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('ohe', ohe_transformer, cat_ohe_cols),
            ('ord', ord_transformer, cat_ord_cols)
        ]
    )

    # 5. 划分训练集与测试集（按目标变量分层抽样，保持类分布比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建包含 SMOTE 和随机森林的模型 Pipeline
    # 使用 imblearn.pipeline 确保 SMOTE 仅在交叉验证的训练折中进行，严防数据泄漏
    full_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 7. 超参数网格设置
    # 注：scikit-learn 中 class_weight 的 '平衡' 对应参数值为 'balanced'
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    # 8. 超参数调优（不适用多进程：n_jobs=1）
    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1
    )

    # 模型拟合
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 9. 模型预测与评估
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 输出最佳参数与评价指标
    print("=" * 40)
    print("最佳超参数组合：")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")

    print("\n测试集评价指标：")
    print(f"  准确率 (Accuracy) : {accuracy:.4f}")
    print(f"  召回率 (Recall)   : {recall:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  F1 分数 (F1-score) : {f1:.4f}")
    print(f"  AUC 曲线下面积    : {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()