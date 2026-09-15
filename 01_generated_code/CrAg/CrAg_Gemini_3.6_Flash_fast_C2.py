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

# 导入 imblearn 管道以防止交叉验证中的数据泄露
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def main():
    # 1. 加载数据集
    file_path = 'CrAg_train.csv'
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"成功加载数据集，数据维度: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到文件 {file_path}，请确保该文件与脚本放置在同一目录下。")

    # 2. 区分特征与目标变量
    target_col = 'CSF-T'
    if target_col not in df.columns:
        # 如果列名有微小差异，可回退到通过最后一列获取
        target_col = df.columns[-1]

    # 二分类转换：目标值 >= 20 为 1，否则为 0
    y = (df[target_col] >= 20).astype(int)
    X = df.drop(columns=[target_col])

    # 3. 按要求划分特征类型
    # 连续变量（数值）
    num_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    # 独热编码分类变量
    ohe_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    # 序数编码分类变量
    ord_cols = ['SER-T', 'Ink staining']

    # 4. 构建数据预处理流水线（严格按照顺序：数值 -> 独热 -> 序数）
    num_transformer = SimpleImputer(strategy='median')
    ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('ohe', ohe_transformer, ohe_cols),
            ('ord', ord_transformer, ord_cols)
        ]
    )

    # 5. 构建完整 Pipeline（防数据泄露核心：SMOTE 仅在每折交叉验证的训练集上运行）
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 6. 设置超参数调优网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 7. 划分训练集与验证集（比例 4:1，采用分层抽样保证类别比例一致）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"训练集样本数: {len(X_train)} | 验证集样本数: {len(X_val)}")

    # 8. 网格搜索调优 (不使用多进程，即 n_jobs=1)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1
    )

    print("开始进行超参数调优与模型训练...")
    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数组合: {grid_search.best_params_}")
    print(f"交叉验证最佳 F1-score: {grid_search.best_score_:.4f}")

    # 9. 在独立验证集上评估模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]

    # 计算各评估指标
    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    # 10. 输出最终评估结果
    print("\n" + "="*40)
    print("      验证集（Validation Set）模型评估结果      ")
    print("="*40)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}")
    print(f"精确率 (Precision) : {prec:.4f}")
    print(f"F1 分数 (F1-score) : {f1:.4f}")
    print(f"ROC AUC 面积       : {auc:.4f}")
    print("="*40)


if __name__ == '__main__':
    main()