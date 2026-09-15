import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


# 1. 加载数据
def load_data(file_path):
    """加载CSV数据文件"""
    df = pd.read_csv(file_path, encoding='utf-8')
    return df


# 2. 定义特征和目标
def prepare_features_target(df):
    """分离特征和目标变量"""
    # 目标列是最后一列CSF-T
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 将目标转换为二分类：是否≥20
    y_binary = (y >= 20).astype(int)

    return X, y_binary


# 3. 定义特征类型
def define_feature_types(X):
    """定义分类变量和连续变量"""
    # 分类变量（包括分类和序数变量）
    categorical_features = ['SER-T', 'Ink staining', 'Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
    # 连续变量
    numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

    # 验证所有特征都存在
    all_features = categorical_features + numeric_features
    missing_features = set(all_features) - set(X.columns)
    if missing_features:
        print(f"警告：以下特征在数据中不存在：{missing_features}")
        # 移除不存在的特征
        categorical_features = [f for f in categorical_features if f in X.columns]
        numeric_features = [f for f in numeric_features if f in X.columns]

    return categorical_features, numeric_features


# 4. 构建预处理流水线
def build_preprocessor(categorical_features, numeric_features):
    """构建数据预处理流水线"""

    # 分类变量的预处理
    # SER-T和Ink staining使用序数编码（假设有序关系）
    ordinal_features = ['SER-T', 'Ink staining']
    # 其他分类变量使用独热编码
    onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    # 过滤出实际存在的特征
    ordinal_features = [f for f in ordinal_features if f in categorical_features]
    onehot_features = [f for f in onehot_features if f in categorical_features]

    # 构建预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), onehot_features),
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ]), numeric_features)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    return preprocessor


# 5. 主流程
def main():
    # 加载数据
    print("加载数据...")
    df = load_data('CrAg_train.csv')
    print(f"数据集形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 准备特征和目标
    X, y = prepare_features_target(df)
    print(f"\n目标变量分布:")
    print(f"  0 (CSF-T < 20): {sum(y == 0)}")
    print(f"  1 (CSF-T >= 20): {sum(y == 1)}")
    print(f"  类别比例: {sum(y == 1) / len(y):.4f}")

    # 定义特征类型
    categorical_features, numeric_features = define_feature_types(X)
    print(f"\n分类变量: {categorical_features}")
    print(f"连续变量: {numeric_features}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 构建预处理流水线
    preprocessor = build_preprocessor(categorical_features, numeric_features)

    # 构建完整的模型流水线（包含SMOTE）
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 设置超参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 网格搜索
    print("\n开始超参数调优...")
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,  # 不使用多进程
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

    # 在测试集上进行预测
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 输出评估结果
    print("\n" + "=" * 50)
    print("模型评估结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score): {f1:.4f}")
    print(f"AUC曲线下面积 (AUC): {auc:.4f}")
    print("=" * 50)

    # 输出混淆矩阵信息
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    return grid_search


if __name__ == "__main__":
    # 运行主程序
    best_model = main()