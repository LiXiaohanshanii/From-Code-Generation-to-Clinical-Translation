import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 加载数据
    data_path = 'CrAg_train.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 2. 构建目标变量：二分类预测 (CSF-T >= 20 为 1，否则为 0)
    # 假设目标列为最后一列（CSF-T）
    target_col = df.columns[-1]
    df['target'] = (df[target_col] >= 20).astype(int)

    # 分离特征和目标变量
    X = df.drop(columns=[target_col, 'target'])
    y = df['target']

    # 3. 明确特征分类
    continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['SER-T', 'Ink staining']

    # 4. 构建数据预处理流（ColumnTransformer）
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量：独热编码 (设置 handle_unknown='ignore' 以防测试集遇到未知类别)
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 序数变量：序数编码 (针对可能存在的缺失值或新类别设置 handle_unknown)
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, continuous_cols),
            ('cat_onehot', onehot_transformer, onehot_cols),
            ('cat_ordinal', ordinal_transformer, ordinal_cols)
        ]
    )

    # 5. 数据集划分 (按 80% 训练集, 20% 测试集划分，按目标变量分层抽样)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 构建包含 SMOTE 和 随机森林 的 Pipeline
    # 注意：必须使用 imblearn.pipeline.Pipeline，确保 SMOTE 只在训练集内部拟合，避免数据渗漏
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. 超参数调优设置
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 使用 GridSearchCV 进行网格搜索，不用多进程 (n_jobs=1)，以 f1 为评分指标
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,
        verbose=1
    )

    print("开始进行模型拟合与超参数调优...")
    grid_search.fit(X_train, y_train)
    print("模型拟合完成！")
    print(f"最佳超参数组合: {grid_search.best_params_}")

    # 8. 使用最佳模型在测试集上评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 9. 输出评估结果
    print("\n" + "=" * 40)
    print("      随机森林分类模型评估结果 (测试集)      ")
    print("=" * 40)
    print(f"准确率 (Accuracy)   : {acc:.4f}")
    print(f"精确率 (Precision)  : {precision:.4f}")
    print(f"召回率 (Recall)     : {recall:.4f}")
    print(f"F1 分数 (F1-score)  : {f1:.4f}")
    print(f"ROC AUC 面积 (AUC)  : {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()