import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# imbalanced-learn 库中的 Pipeline，用于在交叉验证中安全地结合 SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据
    data_path = 'CrAg_train.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 构造二分类目标变量 (CSF-T >= 20 为 1，否则为 0)
    target_col = 'CSF-T'
    X = df.drop(columns=[target_col])
    y = (df[target_col] >= 20).astype(int)

    # 3. 区分变量类型
    continuous_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    ordinal_features = ['SER-T', 'Ink staining']
    onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    # 4. 构建数据预处理流水线 (ColumnTransformer)

    # 连续变量：中位数填充
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 序数分类变量：序数编码（自动适应离散值）
    ordinal_transformer = Pipeline([
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 独热分类变量：独热编码（忽略未知类别）
    onehot_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 组合预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, continuous_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('one', onehot_transformer, onehot_features)
        ]
    )

    # 5. 划分训练集与测试集 (分层抽样，保持类别比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建包含预处理、SMOTE 和随机森林模型的完整 ImbPipeline
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 设置超参数网格（映射至 Pipeline 中的 classifier 步骤）
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 8. 网格搜索超参数调优（使用 5 折交叉验证，不使用多进程 n_jobs=1，以 F1 作为评分指标）
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,
        verbose=1
    )

    print("开始模型超参数调优与训练...")
    grid_search.fit(X_train, y_train)

    print(f"\n最佳模型参数: {grid_search.best_params_}")
    print(f"交叉验证最佳 F1 得分: {grid_search.best_score_:.4f}")

    # 9. 模型评估（使用最佳模型在测试集上预测）
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正例概率用于计算 ROC-AUC

    # 计算各评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 10. 输出评估结果
    print("\n" + "=" * 35)
    print("      测试集模型评估指标")
    print("=" * 35)
    print(f"准确率 (Accuracy)   : {acc:.4f}")
    print(f"召回率 (Recall)     : {rec:.4f}")
    print(f"精确率 (Precision)  : {prec:.4f}")
    print(f"F1 分数 (F1-score)  : {f1:.4f}")
    print(f"AUC 值 (ROC Area)   : {auc:.4f}")
    print("=" * 35)


if __name__ == '__main__':
    main()