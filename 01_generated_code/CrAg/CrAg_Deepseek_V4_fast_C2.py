import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重复
RANDOM_STATE = 42


# 1. 加载数据
def load_data(file_path='CrAg_train.csv'):
    """加载CSV数据文件"""
    data = pd.read_csv(file_path, encoding='utf-8')
    return data


# 2. 数据预处理
def create_preprocessor():
    """
    创建数据预处理器
    包含：
    - 连续变量：中位数填充
    - 分类变量（独热编码）：Color, Transparency, SEX, DEPT, DIAGNOSIS
    - 分类变量（序数编码）：SER-T, Ink staining
    """
    # 定义特征列
    categorical_onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ordinal_cols = ['SER-T', 'Ink staining']
    numerical_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

    # 连续变量处理器：中位数填充
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 独热编码处理器
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 序数编码处理器
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 组合所有预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('onehot', onehot_transformer, categorical_onehot_cols),
            ('ordinal', ordinal_transformer, categorical_ordinal_cols)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    return preprocessor


# 3. 主程序
def main():
    # 加载数据
    print("正在加载数据...")
    data = load_data('CrAg_train.csv')

    # 分离特征和目标变量（CSF-T是最后一列）
    X = data.iloc[:, :-1]  # 所有特征列
    y = data.iloc[:, -1]  # 目标列 CSF-T

    # 将目标变量转换为二分类（是否 >= 20）
    y_binary = (y >= 20).astype(int)

    print(f"数据集形状: {X.shape}")
    print(f"正类样本数 (CSF-T >= 20): {(y_binary == 1).sum()}")
    print(f"负类样本数 (CSF-T < 20): {(y_binary == 0).sum()}")
    print(f"类别比例: {((y_binary == 1).sum() / len(y_binary) * 100):.2f}%")

    # 划分训练集和验证集（4:1）
    print("\n正在划分训练集和验证集...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_binary,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_binary  # 保持类别比例
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")

    # 创建预处理器
    preprocessor = create_preprocessor()

    # 4. 创建包含SMOTE的完整Pipeline（防止数据泄露）
    # 注意：使用imblearn的Pipeline确保SMOTE在交叉验证折内进行
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),  # SMOTE在交叉验证内部进行
        ('classifier', RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1  # 不使用多进程
        ))
    ])

    # 5. 超参数调优
    print("\n正在进行超参数调优...")
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 使用GridSearchCV，评分指标为f1
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # 5折交叉验证
        scoring='f1',
        n_jobs=1,  # 不使用多进程
        verbose=1
    )

    # 训练模型
    print("正在训练模型...")
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

    # 6. 模型评估
    print("\n在验证集上评估模型...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    # 输出评估结果
    print("\n" + "=" * 50)
    print("模型评估结果 (验证集):")
    print("=" * 50)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score): {f1:.4f}")
    print(f"AUC曲线下面积 (AUC-ROC): {auc:.4f}")
    print("=" * 50)

    # 额外信息：特征重要性（可选）
    try:
        # 获取特征名称
        feature_names = []
        # 数值特征
        numerical_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']
        feature_names.extend(numerical_cols)

        # 独热编码特征
        onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
        # 获取独热编码后的特征名
        preprocessor_fitted = best_model.named_steps['preprocessor']
        onehot_transformer = preprocessor_fitted.named_transformers_['onehot']
        onehot_feature_names = onehot_transformer.get_feature_names_out(onehot_cols)
        feature_names.extend(onehot_feature_names)

        # 序数编码特征
        ordinal_cols = ['SER-T', 'Ink staining']
        feature_names.extend(ordinal_cols)

        # 获取特征重要性
        classifier = best_model.named_steps['classifier']
        importances = classifier.feature_importances_

        # 输出前10个重要特征
        feature_importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\n前10个最重要的特征:")
        print(feature_importance_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"\n无法提取特征重要性: {e}")

    return best_model, grid_search.best_params_


if __name__ == "__main__":
    main()