import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')


# 1. 数据加载
def load_data(file_path):
    """
    加载数据文件
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    return df


# 2. 数据预处理
def preprocess_data(df):
    """
    数据预处理：分离特征和目标变量，定义特征类型
    """
    # 目标变量是最后一列
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 将目标变量转换为二分类：是否 >= 16
    y_binary = (y >= 16).astype(int)

    # 定义特征类型
    categorical_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
    categorical_cols_ordinal = ['TPPA']  # 序数编码
    numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

    return X, y_binary, categorical_cols_onehot, categorical_cols_ordinal, numerical_cols


# 3. 构建预处理管道
def build_preprocessor(categorical_cols_onehot, categorical_cols_ordinal, numerical_cols):
    """
    构建数据预处理器
    """
    # 分类变量预处理
    # OneHotEncoder用于独热编码
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # OrdinalEncoder用于序数编码（TPPA是滴度值，可以按顺序编码）
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=[sorted(df['TPPA'].unique())] if 'df' in globals() else None,
                                   handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 连续变量预处理：中位数填充
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', onehot_transformer, categorical_cols_onehot),
            ('ordinal', ordinal_transformer, categorical_cols_ordinal),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='passthrough'  # 保留未指定的列
    )

    return preprocessor


# 4. 构建完整的模型管道
def build_model_pipeline(preprocessor):
    """
    构建包含预处理、SMOTE和随机森林的完整管道
    """
    # 随机森林分类器
    rf_classifier = RandomForestClassifier(
        random_state=42,
        n_jobs=1  # 不使用多进程
    )

    # 创建完整的管道：预处理 -> SMOTE -> 随机森林
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', rf_classifier)
    ])

    return pipeline


# 5. 超参数调优
def tune_hyperparameters(pipeline, X_train, y_train):
    """
    使用GridSearchCV进行超参数调优
    """
    # 定义参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,  # 不使用多进程
        verbose=0
    )

    # 执行超参数搜索
    grid_search.fit(X_train, y_train)

    return grid_search


# 6. 模型评估
def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印结果
    print("=" * 50)
    print("模型评估结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score): {f1:.4f}")
    print(f"AUC值 (Area Under ROC Curve): {auc:.4f}")
    print("=" * 50)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'auc': auc
    }


# 7. 主函数
def main():
    """
    主执行函数
    """
    # 加载数据
    print("正在加载数据...")
    df = load_data('train_data.csv')
    print(f"数据形状: {df.shape}")

    # 预处理数据
    print("\n正在预处理数据...")
    X, y, cat_onehot, cat_ordinal, num_cols = preprocess_data(df)

    # 检查分类变量的缺失值
    print("\n检查分类变量缺失值:")
    all_cat_cols = cat_onehot + cat_ordinal
    for col in all_cat_cols:
        missing = X[col].isnull().sum()
        if missing > 0:
            print(f"警告: {col} 存在 {missing} 个缺失值")
        else:
            print(f"{col}: 无缺失值")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print(f"训练集正例比例: {y_train.mean():.4f}")
    print(f"测试集正例比例: {y_test.mean():.4f}")

    # 构建预处理器
    # 注意：OrdinalEncoder需要预先知道类别，这里我们从训练数据中获取
    print("\n构建预处理器...")
    # 对于TPPA，我们需要知道所有可能的类别值
    tppa_categories = [sorted(X['TPPA'].dropna().unique())]

    # 重新定义序数编码器
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=tppa_categories, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', onehot_transformer, cat_onehot),
            ('ordinal', ordinal_transformer, cat_ordinal),
            ('num', numerical_transformer, num_cols)
        ],
        remainder='passthrough'
    )

    # 构建完整管道
    print("构建模型管道...")
    pipeline = build_model_pipeline(preprocessor)

    # 超参数调优
    print("\n正在进行超参数调优...")
    grid_search = tune_hyperparameters(pipeline, X_train, y_train)

    # 输出最佳参数
    print("\n最佳参数组合:")
    print(grid_search.best_params_)
    print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

    # 使用最佳模型进行预测
    best_model = grid_search.best_estimator_

    # 评估模型
    print("\n评估模型在测试集上的表现:")
    results = evaluate_model(best_model, X_test, y_test)

    # 可选：输出特征重要性（如果需要）
    try:
        # 获取特征名称
        feature_names = []
        # OneHot编码特征
        onehot_encoder = best_model.named_steps['preprocessor'].named_transformers_['onehot']
        # 获取独热编码后的特征名
        for cat_col in cat_onehot:
            # 获取该列的唯一值（除去NaN）
            unique_vals = X_train[cat_col].dropna().unique()
            # 如果使用drop='first'，则去掉第一个类别
            if len(unique_vals) > 1:
                feature_names.extend([f"{cat_col}_{val}" for val in unique_vals[1:]])
            else:
                feature_names.extend([f"{cat_col}_{val}" for val in unique_vals])

        # Ordinal编码特征
        feature_names.extend(cat_ordinal)

        # 数值特征
        feature_names.extend(num_cols)

        # 如果特征数量匹配，则输出重要性
        if len(feature_names) == len(best_model.named_steps['classifier'].feature_importances_):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)
            print("\n特征重要性（Top 10）:")
            print(importance_df.head(10))
    except Exception as e:
        print(f"\n无法显示特征重要性: {e}")

    print("\n模型构建和评估完成！")

    return best_model, results


if __name__ == "__main__":
    # 全局变量用于OrdinalEncoder
    df = None
    main()