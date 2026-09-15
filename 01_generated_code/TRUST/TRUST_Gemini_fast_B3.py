import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# 使用 imblearn 的 Pipeline 避免在交叉验证中发生数据泄露
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def run_random_forest_pipeline(data_path='train_data.csv'):
    # 1. 读取数据
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 构建目标列：TRUST >= 16 为 positive class (1)，< 16 为 negative class (0)
    # 确保 TRUST 列转换为数值类型（处理可能包含字符串的情况）
    df['TRUST_clean'] = pd.to_numeric(df['TRUST'], errors='coerce')
    df['target'] = (df['TRUST_clean'] >= 16).astype(int)

    # 提取特征矩阵与目标向量
    X = df.drop(columns=['TRUST', 'TRUST_clean', 'target'])
    y = df['target']

    # 3. 列名划分
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 数据预处理转换器定义
    # 连续特征中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 独热编码 (SEX, DEPT, DIAGNOSIS)
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 序数编码 (TPPA)
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ]
    )

    # 5. 划分训练集与测试集 (分层抽样保持阳性率一致)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建包含预处理、SMOTE过采样及随机森林的流水线
    # 严格在 Pipeline 中进行 SMOTE，确保训练集与验证集隔离
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
    ])

    # 7. 超参数网格配置（单进程模式）
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 使用 网格搜索 进行调优 (n_jobs=1 不开启多进程)
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,
        verbose=1
    )

    # 8. 训练模型
    print("开始模型训练与参数搜索...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n最佳参数组合: {grid_search.best_params_}")

    # 9. 测试集评估
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 10. 打印评估结果
    print("\n" + "=" * 45)
    print("随机森林模型评估结果 (测试集):")
    print("=" * 45)
    print(f"准确率   (Accuracy):  {acc:.4f}")
    print(f"召回率   (Recall):    {rec:.4f}")
    print(f"精确率   (Precision): {prec:.4f}")
    print(f"F1 分数  (F1-score):  {f1:.4f}")
    print(f"AUC 面积 (ROC-AUC):   {auc:.4f}")
    print("=" * 45)

    return best_model


if __name__ == '__main__':
    run_random_forest_pipeline()