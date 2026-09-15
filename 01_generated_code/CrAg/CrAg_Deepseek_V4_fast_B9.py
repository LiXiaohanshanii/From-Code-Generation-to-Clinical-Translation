import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')


# 1. 读取数据
def load_data(file_path):
    """加载CSV数据"""
    df = pd.read_csv(file_path, encoding='utf-8')
    return df


# 2. 数据预处理
def preprocess_data(df):
    """数据预处理：特征分离、缺失值处理、编码准备"""

    # 分离特征和目标变量（CSF-T是最后一列）
    X = df.iloc[:, :-1]  # 所有特征列
    y = df.iloc[:, -1]  # 目标列 CSF-T

    # 将目标变量转换为二分类（>=20 为1，<20为0）
    y_binary = (y >= 20).astype(int)

    # 定义特征类型
    categorical_cols_onehot = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    categorical_cols_ordinal = ['SER-T', 'Ink staining']
    numerical_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

    # 检查列是否存在
    # 只保留实际存在的列
    categorical_cols_onehot = [col for col in categorical_cols_onehot if col in X.columns]
    categorical_cols_ordinal = [col for col in categorical_cols_ordinal if col in X.columns]
    numerical_cols = [col for col in numerical_cols if col in X.columns]

    # 验证特征完整性
    all_features = categorical_cols_onehot + categorical_cols_ordinal + numerical_cols
    missing_features = set(X.columns) - set(all_features)
    if missing_features:
        print(f"警告：以下特征未在预定义列表中：{missing_features}")

    return X, y_binary, categorical_cols_onehot, categorical_cols_ordinal, numerical_cols


# 3. 构建预处理流水线
def build_preprocessor(cat_onehot, cat_ordinal, num_cols):
    """构建数据预处理流水线"""

    # 数值特征：中位数填充
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 分类特征（独热编码）
    categorical_onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 分类特征（序数编码）- 使用OrdinalEncoder
    categorical_ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 组合所有预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_cols),
            ('cat_onehot', categorical_onehot_transformer, cat_onehot),
            ('cat_ordinal', categorical_ordinal_transformer, cat_ordinal)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    return preprocessor


# 4. 构建完整模型流水线（包含SMOTE和分类器）
def build_model_pipeline(preprocessor):
    """构建包含SMOTE和随机森林的完整流水线"""

    # 随机森林分类器
    rf_classifier = RandomForestClassifier(
        random_state=42,
        n_jobs=1  # 不使用多进程
    )

    # 创建包含SMOTE的流水线
    # 注意：SMOTE应该在预处理之后、分类器之前应用
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', rf_classifier)
    ])

    return pipeline


# 5. 超参数调优
def hyperparameter_tuning(pipeline, X_train, y_train):
    """使用网格搜索进行超参数调优"""

    # 定义参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 网格搜索
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # 5折交叉验证
        scoring='f1',
        n_jobs=1,  # 不使用多进程
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

    return grid_search


# 6. 模型评估
def evaluate_model(model, X_test, y_test):
    """评估模型性能"""

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
    print("\n" + "=" * 50)
    print("模型评估结果：")
    print("=" * 50)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score): {f1:.4f}")
    print(f"AUC曲线下面积 (AUC): {auc:.4f}")
    print("=" * 50)

    # 返回评估指标字典
    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc': auc
    }

    return metrics


# 7. 主函数
def main():
    """主执行函数"""

    # 数据文件路径
    file_path = 'CrAg_train.csv'

    try:
        # 1. 加载数据
        print("正在加载数据...")
        df = load_data(file_path)
        print(f"数据集形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        # 2. 预处理数据
        print("\n正在预处理数据...")
        X, y, cat_onehot, cat_ordinal, num_cols = preprocess_data(df)

        # 检查目标变量分布
        print(f"\n目标变量分布（>=20为1，<20为0）：")
        print(y.value_counts())

        # 3. 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

        # 4. 构建预处理流水线
        print("\n构建预处理流水线...")
        preprocessor = build_preprocessor(cat_onehot, cat_ordinal, num_cols)

        # 5. 构建完整模型流水线
        print("构建完整模型流水线...")
        pipeline = build_model_pipeline(preprocessor)

        # 6. 超参数调优
        print("\n开始超参数调优...")
        best_model = hyperparameter_tuning(pipeline, X_train, y_train)

        # 7. 评估模型
        print("\n评估最佳模型...")
        metrics = evaluate_model(best_model, X_test, y_test)

        print("\n模型训练和评估完成！")

        return best_model, metrics

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到，请确保文件在当前目录下。")
    except Exception as e:
        print(f"发生错误：{e}")
        import traceback
        traceback.print_exc()


# 执行主函数
if __name__ == "__main__":
    best_model, metrics = main()