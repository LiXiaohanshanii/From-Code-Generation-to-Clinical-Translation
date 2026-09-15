import pandas as pd
import numpy as np

# 导入 Scikit-Learn 模块
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# 导入 imbalanced-learn 管道与 SMOTE（防止交叉验证过程中的数据泄漏）
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # ---------------------------------------------------------
    # 1. 数据加载与目标变量预处理
    # ---------------------------------------------------------
    print(">>> 正在加载数据集 train_data.csv ...")
    data_path = 'train_data.csv'
    df = pd.read_csv(data_path, encoding='utf-8')

    # 提取目标列 (TRUST 位于最后一列)
    target_col = df.columns[-1]

    # 转换为二分类任务：TRUST >= 16 记为 1（强阳性/高滴度），< 16 记为 0
    df['target'] = (df[target_col] >= 16).astype(int)

    # 划分特征矩阵 X 与目标向量 y
    X = df.drop(columns=[target_col, 'target'])
    y = df['target']

    # ---------------------------------------------------------
    # 2. 定义特征分类与预处理流水线 (ColumnTransformer)
    # ---------------------------------------------------------
    # 分类变量分类定义
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
    ordinal_cols = ['TPPA']  # 序数编码

    # 连续变量定义
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 构建列转换器
    preprocessor = ColumnTransformer(
        transformers=[
            # 连续变量：中位数填充
            ('num_imputer', SimpleImputer(strategy='median'), num_cols),

            # 无序分类变量：独热编码 (设置 handle_unknown='ignore' 应对未见过的类别)
            ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),

            # 有序分类变量：序数编码 (未见过的类别默认编码为 -1)
            ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols)
        ]
    )

    # ---------------------------------------------------------
    # 3. 划分训练集与测试集
    # ---------------------------------------------------------
    # 使用分层抽样 (stratify=y) 保证训练集与测试集中正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # 4. 构建包含 SMOTE 和随机森林的流水线 (ImbPipeline)
    # ---------------------------------------------------------
    # 注意：必须使用 imblearn 的 Pipeline，保证 SMOTE 仅在每折交叉验证的训练子集上施加
    full_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ---------------------------------------------------------
    # 5. 网格搜索超参数调优 (GridSearchCV)
    # ---------------------------------------------------------
    # 参数网格设置（匹配要求中的超参数范围，'平衡' 对应 sklearn 中的 'balanced'）
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print(">>> 正在启动网格搜索超参数调优（不开启多进程 n_jobs=1）...")
    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring='roc_auc',  # 以 ROC-AUC 指标选择最优参数
        n_jobs=1  # 不开启多进程
    )

    grid_search.fit(X_train, y_train)

    print("\n[调优完成] 最佳超参数组合为:")
    for param_name, val in grid_search.best_params_.items():
        print(f"  - {param_name.split('__')[1]}: {val}")

    # ---------------------------------------------------------
    # 6. 模型评估
    # ---------------------------------------------------------
    best_model = grid_search.best_estimator_

    # 在独立测试集上进行预测
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估报告
    print("\n" + "=" * 50)
    print("               随机森林分类模型评估指标               ")
    print("=" * 50)
    print(f" 准确率 (Accuracy)   : {acc:.4f}")
    print(f" 召回率 (Recall)     : {rec:.4f}")
    print(f" 精确率 (Precision)  : {prec:.4f}")
    print(f" F1 分数 (F1-score)  : {f1:.4f}")
    print(f" AUC 曲线下面积      : {auc:.4f}")
    print("=" * 50)

    print("\n详细分类报告 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['TRUST < 16', 'TRUST >= 16']))


if __name__ == '__main__':
    main()