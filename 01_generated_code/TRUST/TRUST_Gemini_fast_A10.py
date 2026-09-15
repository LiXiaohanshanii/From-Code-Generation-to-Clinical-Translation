import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 构建目标变量（二分类任务：TRUST >= 16 为 1，否则为 0）
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # 定义变量类型列表
    onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['TPPA']
    numeric_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 3. 构建预处理流水线 (ColumnTransformer)
    # - 连续变量：中位数填充
    # - 独热编码变量：OneHotEncoder
    # - 序数编码变量：OrdinalEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat_onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
            ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
        ]
    )

    # 4. 数据预处理转换
    X_processed = preprocessor.fit_transform(X)

    # 5. 划分训练集和测试集 (80% 训练, 20% 测试，保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 使用 SMOTE 处理训练集类别不平衡（仅对训练集操作，避免数据泄露）
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 7. 定义随机森林网格搜索参数
    # 注：class_weight 在 sklearn 中对应 'balanced'，此处照常适配
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }

    rf = RandomForestClassifier(random_state=42)

    # 不使用多进程 (n_jobs=1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    print("开始模型超参数调优...")
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_model = grid_search.best_estimator_
    print(f"最佳超参数组合: {grid_search.best_params_}\n")

    # 8. 在测试集上进行预测与评估
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各项指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 9. 输出评估结果
    print("====== 模型评估结果 ======")
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"F1 分数 (F1-score): {f1:.4f}")
    print(f"AUC 值 (ROC AUC)  : {auc:.4f}")


if __name__ == '__main__':
    main()