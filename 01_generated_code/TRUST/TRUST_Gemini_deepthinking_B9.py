import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE


def build_and_evaluate_rf():
    # 1. 加载数据集
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构建目标变量：判断 TRUST 是否 >= 16 (二分类任务)
    target_col = 'TRUST'
    y = (df[target_col] >= 16).astype(int)
    X = df.drop(columns=[target_col])

    # 3. 明确特征分类
    ohe_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['TPPA']
    continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 构建预处理管道 (ColumnTransformer)
    # 连续变量：中位数填充；分类变量：独热编码与序数编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), continuous_features),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_features),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
        ],
        remainder='drop'
    )

    # 5. 划分训练集与测试集 (80% 训练，20% 测试)，采用分层抽样保持类别比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 对训练集和测试集施加特征转换（先拟合预处理，防止数据泄漏）
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # 7. 使用 SMOTE 对训练集进行过采样（平衡正负样本）
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

    # 8. 设置随机森林超参数网格（单进程执行 n_jobs=1）
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=1)

    # 使用网格搜索进行超参数调优（不使用多进程 n_jobs=1）
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    # 拟合重采样后的训练数据
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_rf = grid_search.best_estimator_

    # 9. 模型预测与评价
    y_pred = best_rf.predict(X_test_preprocessed)
    y_prob = best_rf.predict_proba(X_test_preprocessed)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 10. 输出评估结果
    print("=" * 45)
    print("最佳超参数:", grid_search.best_params_)
    print("=" * 45)
    print(f"准确率 (Accuracy)   : {acc:.4f}")
    print(f"召回率 (Recall)     : {rec:.4f}")
    print(f"精确率 (Precision)  : {prec:.4f}")
    print(f"F1 分数 (F1-score)  : {f1:.4f}")
    print(f"ROC-AUC 面积 (AUC)  : {auc:.4f}")
    print("=" * 45)


if __name__ == '__main__':
    build_and_evaluate_rf()