import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# 如果没有安装 imbalanced-learn，请在终端执行: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE


def main():
    # 1. 读取数据集
    file_path = 'CrAg_train.csv'
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"错误: 未找到文件 {file_path}，请确保该文件位于当前工作目录下。")
        return

    # 2. 目标变量处理：判断 CSF-T 是否 >= 20 (二分类任务)
    target_col = 'CSF-T'
    if target_col not in df.columns:
        # 如果最后一列列名不是 'CSF-T'，直接按位置取最后一列
        target_col = df.columns[-1]

    # 构建二分类目标：>= 20 为 1，否则为 0
    y = (df[target_col] >= 20).astype(int)
    X = df.drop(columns=[target_col])

    # 3. 定义特征分类
    # 连续变量
    continuous_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
    # 序数编码分类变量
    ordinal_features = ['SER-T', 'Ink staining']
    # 独热编码分类变量
    onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    # 4. 构建预处理流水线 (ColumnTransformer)
    # 4.1 连续变量处理：中位数填充
    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 4.2 序数分类变量处理
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 4.3 独热分类变量处理
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 组合各个预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', continuous_transformer, continuous_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('cat', onehot_transformer, onehot_features)
        ]
    )

    # 5. 数据集划分 (80% 训练集, 20% 测试集)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 特征转换
    # 在训练集上拟合预处理，并转换训练集与测试集
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 7. 分类不平衡处理：使用 SMOTE（仅应用于训练集）
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

    # 8. 随机森林分类器与网格搜索调优
    rf = RandomForestClassifier(random_state=42)

    # 设定指定的超参数网格
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }

    # 使用 GridSearchCV 进行超参数调优，不使用多进程 (n_jobs=1)，评分指标为 f1
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1
    )

    # 模型拟合
    grid_search.fit(X_train_resampled, y_train_resampled)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 9. 模型评估（在测试集上预测）
    y_pred = best_model.predict(X_test_transformed)
    y_pred_proba = best_model.predict_proba(X_test_transformed)[:, 1]

    # 计算各项评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 10. 输出评估结果
    print("=" * 40)
    print("      最佳模型超参数      ")
    print("=" * 40)
    for param, val in grid_search.best_params_.items():
        print(f"{param}: {val}")

    print("\n" + "=" * 40)
    print("     随机森林模型评估指标 (测试集)     ")
    print("=" * 40)
    print(f"准确率   (Accuracy)  : {acc:.4f}")
    print(f"召回率   (Recall)    : {rec:.4f}")
    print(f"精确率   (Precision) : {prec:.4f}")
    print(f"F1 分数  (F1-score)  : {f1:.4f}")
    print(f"ROC-AUC  (Area Under ROC): {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()