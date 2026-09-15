import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 加载数据
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构造二分类目标变量 (>=16 记为 1, <16 记为 0)
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # 3. 定义特征分类
    onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['TPPA']
    numeric_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 特征预处理管道（ColumnTransformer）
    # - 连续变量：中位数填充
    # - 独热编码变量：OneHotEncoder
    # - 序数编码变量：OrdinalEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
        ]
    )

    # 5. 划分训练集和测试集 (采用分层抽样保持标签分布)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 使用 imblearn Pipeline 整合预处理、SMOTE 和随机森林模型
    # 注意：为了防止数据泄露，SMOTE 只在训练集划分内进行（由 Pipeline 自动处理）
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 定义超参数网格
    # 注意：Pipeline 中访问模型参数需加上 'classifier__' 前缀
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']  # 对应要求中的 'balanced'
    }

    # 8. 网格搜索调优 (禁用多进程：n_jobs=1)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    print("正在进行模型训练与超参数调优...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳超参数配置: {grid_search.best_params_}")

    # 9. 模型预测与评估
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 10. 输出评估结果
    print("\n========= 模型评估结果 =========")
    print(f"准确率 (Accuracy) : {accuracy:.4f}")
    print(f"召回率 (Recall)   : {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1 分数 (F1-score) : {f1:.4f}")
    print(f"ROC-AUC 曲线下面积 : {auc:.4f}")
    print("==================================")


if __name__ == '__main__':
    main()