import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# 引入 imblearn 的 Pipeline 以确保 SMOTE 仅在交叉验证的训练折中进行
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def main():
    # 1. 加载数据集
    file_path = 'train_data.csv'
    df = pd.read_csv(file_path, encoding='utf-8')

    # 2. 构建二分类目标变量 (TRUST >= 16 为 1，否则为 0)
    X = df.drop(columns=['TRUST'])
    y = (df['TRUST'] >= 16).astype(int)

    # 定义特征分类
    cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ordinal_cols = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 3. 划分数据集 (保留类别比例分层抽样)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 构建数据预处理流水线
    # 连续变量：中位数填充
    # 分类变量：One-Hot 编码 / Ordinal 编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_onehot_cols),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_ordinal_cols)
        ]
    )

    # 5. 构建完整 Pipeline (预处理 -> SMOTE -> 随机森林分类器)
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 6. 定义超参数搜索空间 (注意：'平衡' 对应 sklearn 中的 'balanced')
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10],
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced']
    }

    # 7. 超参数调优 (GridSearchCV，不使用多进程 n_jobs=1)
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    print("开始模型拟合与超参数调优...")
    grid_search.fit(X_train, y_train)

    # 8. 最佳模型获取与预测
    best_model = grid_search.best_estimator_
    print(f"\n最佳参数组合: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 9. 模型评估指标计算
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 10. 输出评估结果
    print("\n" + "="*40)
    print("      测试集模型评估结果 (TRUST >= 16)")
    print("="*40)
    print(f"准确率 (Accuracy)   : {acc:.4f}")
    print(f"召回率 (Recall)     : {recall:.4f}")
    print(f"精确率 (Precision)  : {precision:.4f}")
    print(f"F1 分数 (F1-score)  : {f1:.4f}")
    print(f"ROC AUC             : {auc:.4f}")
    print("="*40)

if __name__ == '__main__':
    main()