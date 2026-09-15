import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE


def main():
    # 1. 加载数据
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 处理拼写兼容性（避免列名中 DIAGNOSIS 与 DIAGONSIS 的拼写差异）
    if 'DIAGONSIS' in df.columns and 'DIAGNOSIS' not in df.columns:
        df.rename(columns={'DIAGONSIS': 'DIAGNOSIS'}, inplace=True)

    # 2. 构建目标列：TRUST 滴度 >= 16 为正例(1)，否则为负例(0)
    target_col = 'TRUST'
    X = df.drop(columns=[target_col])
    y = (df[target_col] >= 16).astype(int)

    # 声明特征类型
    cat_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ord_cols = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 3. 划分训练集与测试集（按标签比例分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 构建预处理流水线
    # - 连续变量：中位数填充
    # - 分类变量：One-Hot 编码 / 序数编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_ohe_cols),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_ord_cols)
        ]
    )

    # 在训练集上拟合预处理器，并转换训练集与测试集
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # 5. 使用 SMOTE 进行类别不平衡处理（仅对训练集操作）
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)

    # 6. 超参数调优设置 (网格搜索，单进程 n_jobs=1)
    # sklearn 中 'class_weight': 'balanced' 对应中文设置中的 '平衡'
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }

    rf_base = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1
    )

    # 拟合模型
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_rf = grid_search.best_estimator_

    # 7. 测试集预测与指标评估
    y_pred = best_rf.predict(X_test_encoded)
    y_pred_proba = best_rf.predict_proba(X_test_encoded)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 8. 打印结果
    print("=" * 40)
    print("最佳超参数组合：")
    print(grid_search.best_params_)
    print("=" * 40)
    print("模型评估结果（测试集）：")
    print(f"准确率   (Accuracy)  : {accuracy:.4f}")
    print(f"精确率   (Precision) : {precision:.4f}")
    print(f"召回率   (Recall)    : {recall:.4f}")
    print(f"F1分数   (F1-score)  : {f1:.4f}")
    print(f"ROC-AUC (Area Under) : {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()