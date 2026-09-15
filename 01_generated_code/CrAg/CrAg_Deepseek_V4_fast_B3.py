# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')


def main():
    # 1. 读取数据
    print("正在读取数据...")
    data = pd.read_csv('CrAg_train.csv', encoding='utf-8')

    # 2. 分离特征和目标变量
    # 目标列是最后一列 'CSF-T'
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 将目标变量转换为二分类：是否 >= 20
    y_binary = (y >= 20).astype(int)

    print(f"数据集形状: {X.shape}")
    print(f"正样本比例 (>=20): {y_binary.mean():.4f}")

    # 3. 定义特征类型
    # 分类变量（独热编码）
    categorical_onehot = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    # 分类变量（序数编码）
    categorical_ordinal = ['SER-T', 'Ink staining']

    # 连续变量（需要中位数填充）
    numerical_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

    # 检查特征是否都存在
    all_features = categorical_onehot + categorical_ordinal + numerical_features
    missing_features = [f for f in all_features if f not in X.columns]
    if missing_features:
        print(f"警告：以下特征在数据中不存在: {missing_features}")
        # 移除不存在的特征
        categorical_onehot = [f for f in categorical_onehot if f in X.columns]
        categorical_ordinal = [f for f in categorical_ordinal if f in X.columns]
        numerical_features = [f for f in numerical_features if f in X.columns]

    print(f"独热编码特征: {categorical_onehot}")
    print(f"序数编码特征: {categorical_ordinal}")
    print(f"数值特征: {numerical_features}")

    # 4. 数据划分（先划分再处理，避免数据泄露）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print(f"训练集正样本比例: {y_train.mean():.4f}")
    print(f"测试集正样本比例: {y_test.mean():.4f}")

    # 5. 构建预处理步骤
    # 5.1 数值特征预处理：中位数填充
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 5.2 独热编码（处理未知类别）
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 5.3 序数编码
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 5.4 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('onehot', onehot_transformer, categorical_onehot),
            ('ordinal', ordinal_transformer, categorical_ordinal)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    # 6. 构建包含SMOTE的完整流水线
    # 注意：SMOTE应该在预处理之后、模型训练之前应用
    # 但由于SMOTE需要数值数据，我们在预处理之后应用SMOTE

    # 先创建一个只包含预处理的流水线来转换数据
    print("正在预处理数据...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # 获取特征名称（用于调试）
    feature_names = []
    # 数值特征
    feature_names.extend(numerical_features)
    # 独热编码特征
    for cat in categorical_onehot:
        if cat in X_train.columns:
            unique_vals = X_train[cat].dropna().unique()
            feature_names.extend([f"{cat}_{val}" for val in sorted(unique_vals)])
    # 序数编码特征
    feature_names.extend(categorical_ordinal)

    print(f"预处理后特征数量: {X_train_preprocessed.shape[1]}")

    # 7. 应用SMOTE
    print("正在应用SMOTE处理不平衡数据...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

    print(f"SMOTE后训练集大小: {X_train_resampled.shape[0]}")
    print(f"SMOTE后正样本比例: {y_train_resampled.mean():.4f}")

    # 8. 定义随机森林模型和超参数网格
    rf = RandomForestClassifier(random_state=42, n_jobs=1)  # 不使用多进程

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }

    # 9. 网格搜索调优
    print("正在进行超参数调优（以F1为评分指标）...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,  # 不使用多进程
        verbose=1
    )

    grid_search.fit(X_train_resampled, y_train_resampled)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

    # 10. 使用最佳模型进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_preprocessed)
    y_pred_proba = best_model.predict_proba(X_test_preprocessed)[:, 1]

    # 11. 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 12. 输出评估结果
    print("\n" + "=" * 50)
    print("模型评估结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score):  {f1:.4f}")
    print(f"AUC曲线下面积:      {auc:.4f}")
    print("=" * 50)

    # 13. 特征重要性（可选）
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        # 打印Top10重要特征
        top_indices = np.argsort(importances)[-10:][::-1]
        print("\nTop 10 重要特征:")
        for idx in top_indices:
            if idx < len(feature_names):
                print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

    return best_model, preprocessor, grid_search.best_params_


if __name__ == "__main__":
    # 设置随机种子确保可重复性
    np.random.seed(42)
    model, preprocessor, best_params = main()