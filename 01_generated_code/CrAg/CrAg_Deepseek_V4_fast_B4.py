import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')


# 1. 加载数据
def load_data(file_path):
    """加载CSV数据"""
    df = pd.read_csv(file_path, encoding='utf-8')
    return df


# 2. 定义特征和目标
def prepare_features_target(df):
    """分离特征和目标变量"""
    # 目标列是最后一列 CSF-T
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 将目标转换为二分类：是否 >= 20
    y_binary = (y >= 20).astype(int)

    return X, y_binary


# 3. 定义特征类型
def define_feature_types(X):
    """定义分类变量和连续变量"""
    # 分类变量
    categorical_cols = ['SER-T', 'Ink staining', 'Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    # 连续变量
    numerical_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

    # 验证所有列都在数据中
    for col in categorical_cols + numerical_cols:
        if col not in X.columns:
            print(f"警告: 列 '{col}' 不在数据中")

    # 只保留实际存在的列
    categorical_cols = [col for col in categorical_cols if col in X.columns]
    numerical_cols = [col for col in numerical_cols if col in X.columns]

    return categorical_cols, numerical_cols


# 4. 创建预处理管道
def create_preprocessor(categorical_cols, numerical_cols):
    """创建数据预处理管道"""

    # 分类变量处理：
    # - SER-T, Ink staining: 序数编码 (OrdinalEncoder)
    # - Color, Transparency, SEX, DEPT, DIAGNOSIS: 独热编码 (OneHotEncoder)

    ordinal_cols = ['SER-T', 'Ink staining']
    onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    # 只保留实际存在的列
    ordinal_cols = [col for col in ordinal_cols if col in categorical_cols]
    onehot_cols = [col for col in onehot_cols if col in categorical_cols]

    # 数值变量：中位数填充
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 序数编码
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 独热编码
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 组合所有预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('ord', ordinal_transformer, ordinal_cols),
            ('onehot', onehot_transformer, onehot_cols)
        ],
        remainder='drop'
    )

    return preprocessor


# 5. 构建完整的管道
def build_pipeline(preprocessor):
    """构建包含预处理、SMOTE和随机森林的完整管道"""

    # 随机森林分类器
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=1  # 不使用多进程
    )

    # 使用imblearn的Pipeline以支持SMOTE
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', rf)
    ])

    return pipeline


# 6. 定义超参数网格
def get_param_grid():
    """定义超参数搜索网格"""
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }
    return param_grid


# 7. 主执行流程
def main():
    # 数据文件路径
    file_path = 'CrAg_train.csv'

    print("=" * 60)
    print("随机森林分类模型构建与评估")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    df = load_data(file_path)
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    # 2. 准备特征和目标
    print("\n[2] 准备特征和目标变量...")
    X, y = prepare_features_target(df)
    print(f"特征形状: {X.shape}")
    print(f"目标分布: \n{y.value_counts().to_dict()}")
    print(f"目标比例 (1类占比): {y.mean():.2%}")

    # 3. 划分训练集和测试集
    print("\n[3] 划分训练集和测试集 (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")

    # 4. 定义特征类型
    print("\n[4] 定义特征类型...")
    categorical_cols, numerical_cols = define_feature_types(X)
    print(f"分类变量: {categorical_cols}")
    print(f"连续变量: {numerical_cols}")

    # 5. 创建预处理管道
    print("\n[5] 创建预处理管道...")
    preprocessor = create_preprocessor(categorical_cols, numerical_cols)

    # 6. 构建完整管道
    print("\n[6] 构建完整管道 (预处理 + SMOTE + 随机森林)...")
    pipeline = build_pipeline(preprocessor)

    # 7. 超参数调优
    print("\n[7] 超参数调优 (GridSearchCV)...")
    param_grid = get_param_grid()

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,  # 不使用多进程
        verbose=1
    )

    print("开始网格搜索...")
    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

    # 8. 在测试集上评估
    print("\n[8] 在测试集上评估模型...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 输出结果
    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"准确率 (Accuracy):   {accuracy:.4f}")
    print(f"召回率 (Recall):     {recall:.4f}")
    print(f"精确率 (Precision):  {precision:.4f}")
    print(f"F1分数 (F1-score):   {f1:.4f}")
    print(f"AUC值 (ROC-AUC):     {auc:.4f}")
    print("=" * 60)

    # 9. 特征重要性（可选，但不进行可视化）
    try:
        # 获取随机森林模型
        rf_model = best_model.named_steps['classifier']

        # 获取特征名称（经过OneHot编码后）
        preprocessor_fitted = best_model.named_steps['preprocessor']

        # 获取数值特征名称
        num_features = numerical_cols

        # 获取序数编码特征名称
        ordinal_cols = ['SER-T', 'Ink staining']
        ordinal_cols = [col for col in ordinal_cols if col in categorical_cols]

        # 获取独热编码特征名称
        onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
        onehot_cols = [col for col in onehot_cols if col in categorical_cols]

        # 获取OneHotEncoder的类别名称
        onehot_encoder = preprocessor_fitted.named_transformers_['onehot'].named_steps['onehot']
        onehot_feature_names = []
        for col in onehot_cols:
            # 获取该列在ColumnTransformer中的位置
            # 由于ColumnTransformer可能改变顺序，我们需要通过名称获取
            # 简化处理：直接获取编码后的特征名
            if hasattr(onehot_encoder, 'get_feature_names_out'):
                # 对于OneHotEncoder，获取特征名
                categories = onehot_encoder.categories_
                # 需要获取对应列的位置
                # 这里简化处理
                pass

        # 获取所有特征名（简化版本）
        feature_names = num_features + ordinal_cols
        # 为OneHot编码添加列名
        for col in onehot_cols:
            # 简单处理：用列名加类别值
            if col in X.columns:
                unique_vals = X[col].dropna().unique()
                for val in sorted(unique_vals):
                    if pd.notna(val):
                        feature_names.append(f"{col}_{val}")

        # 确保特征重要性长度与特征名匹配
        importances = rf_model.feature_importances_
        if len(importances) == len(feature_names):
            # 按重要性排序（不打印详细列表，只打印top5）
            feature_importance = sorted(zip(feature_names, importances),
                                        key=lambda x: x[1], reverse=True)
            print("\n特征重要性 (Top 5):")
            for name, imp in feature_importance[:5]:
                print(f"  {name}: {imp:.4f}")
        else:
            print(f"\n特征重要性长度({len(importances)})与特征名长度({len(feature_names)})不匹配，跳过显示")

    except Exception as e:
        print(f"\n无法获取特征重要性: {e}")

    # 保存模型（可选）
    # import joblib
    # joblib.dump(best_model, 'random_forest_model.pkl')
    # print("\n模型已保存为 'random_forest_model.pkl'")

    return best_model, grid_search


if __name__ == "__main__":
    best_model, grid_search = main()