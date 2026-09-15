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

from imblearn.over_sampling import SMOTE


def main():
    # ---------------------------------------------------------
    # 1. 数据加载与目标变量处理
    # ---------------------------------------------------------
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 定义特征变量与目标变量 (TRUST >= 16 定义为阳性类 1，否则为 0)
    X = df.drop(columns=['TRUST'])
    y = (df['TRUST'] >= 16).astype(int)

    # ---------------------------------------------------------
    # 2. 特征分类定义
    # ---------------------------------------------------------
    categorical_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ordinal = ['TPPA']
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # ---------------------------------------------------------
    # 3. 构建数据预处理流水线 (ColumnTransformer)
    # ---------------------------------------------------------
    # 连续变量：中位数填充
    # 分类变量：独热编码 / 序数编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), continuous_cols),
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_onehot),
            ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_ordinal)
        ],
        remainder='drop'
    )

    # ---------------------------------------------------------
    # 4. 数据集拆分 (训练集 与 测试集)
    # ---------------------------------------------------------
    # 分层抽样 (stratify=y)，保证训练集与测试集中目标类的比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # 5. 特征预处理拟合与转换
    # ---------------------------------------------------------
    # 在训练集上拟合预处理逻辑，并应用到训练集和测试集
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # ---------------------------------------------------------
    # 6. 处理类别不平衡 (仅对训练集使用 SMOTE)
    # ---------------------------------------------------------
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

    # ---------------------------------------------------------
    # 7. 随机森林模型构建与网格搜索调参
    # ---------------------------------------------------------
    rf_clf = RandomForestClassifier(random_state=42)

    # 参数网格定义 (注: sklearn 中 class_weight='balanced' 对应中文“平衡”)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }

    # 禁用多进程 n_jobs=1，保证 PyCharm 调试及运行稳定性
    grid_search = GridSearchCV(
        estimator=rf_clf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=1
    )

    # 在过采样后的训练集上拟合网格搜索
    grid_search.fit(X_train_res, y_train_res)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print("\n[网格搜索] 最佳超参数组合：", grid_search.best_params_)

    # ---------------------------------------------------------
    # 8. 模型预测与评估
    # ---------------------------------------------------------
    y_pred = best_model.predict(X_test_proc)
    y_prob = best_model.predict_proba(X_test_proc)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 输出评估结果
    print("\n" + "=" * 40)
    print("        模型评估指标 (测试集)        ")
    print("=" * 40)
    print(f"准确率 (Accuracy)   : {accuracy:.4f}")
    print(f"召回率 (Recall)     : {recall:.4f}")
    print(f"精确率 (Precision)  : {precision:.4f}")
    print(f"F1 分数 (F1-score)  : {f1:.4f}")
    print(f"ROC-AUC 面积 (AUC)  : {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()