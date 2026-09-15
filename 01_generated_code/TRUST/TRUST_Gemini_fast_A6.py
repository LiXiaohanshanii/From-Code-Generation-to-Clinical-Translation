import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
    # 1. 读取数据与构建目标变量
    # ---------------------------------------------------------
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 构建二分类目标变量 (TRUST >= 16 为 1，否则为 0)
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # ---------------------------------------------------------
    # 2. 定义特征分类
    # ---------------------------------------------------------
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']

    # ---------------------------------------------------------
    # 3. 数据集划分 (按目标变量分层抽样)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # 4. 构建数据预处理 Pipeline
    # ---------------------------------------------------------
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 独热编码分类变量
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 序数编码分类变量
    ordinal_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, continuous_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ],
        remainder='drop'
    )

    # 对训练集和测试集分别进行预处理转换
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # ---------------------------------------------------------
    # 5. SMOTE 类别不平衡处理 (仅作用于训练集)
    # ---------------------------------------------------------
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)

    # ---------------------------------------------------------
    # 6. 随机森林模型构建与网格搜索调优 (不使用多进程 n_jobs=1)
    # ---------------------------------------------------------
    rf = RandomForestClassifier(random_state=42, n_jobs=1)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,  # 禁用多进程
        verbose=1
    )

    grid_search.fit(X_train_res, y_train_res)
    best_model = grid_search.best_estimator_

    print("\n最优模型参数：", grid_search.best_params_)

    # ---------------------------------------------------------
    # 7. 模型预测与评估
    # ---------------------------------------------------------
    y_pred = best_model.predict(X_test_prep)
    y_pred_proba = best_model.predict_proba(X_test_prep)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 输出评估结果
    print("\n========== 测试集模型评估结果 ==========")
    print(f"准确率 (Accuracy)   : {acc:.4f}")
    print(f"召回率 (Recall)     : {rec:.4f}")
    print(f"精确率 (Precision)  : {prec:.4f}")
    print(f"F1 分数 (F1-score)  : {f1:.4f}")
    print(f"ROC-AUC 面积 (AUC)  : {auc:.4f}")
    print("========================================")


if __name__ == '__main__':
    main()