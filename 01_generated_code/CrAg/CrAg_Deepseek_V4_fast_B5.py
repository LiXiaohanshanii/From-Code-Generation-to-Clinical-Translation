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


# 1. 数据加载
def load_data(file_path):
    """加载CSV数据"""
    df = pd.read_csv(file_path, encoding='utf-8')
    return df


# 2. 数据预处理
def preprocess_data(df):
    """数据预处理：分离特征和目标，定义特征类型"""

    # 目标列是最后一列CSF-T
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 将目标转换为二分类：是否≥20
    y_binary = (y >= 20).astype(int)

    # 定义特征类型
    # 分类变量（有序编码）
    ordinal_features = ['SER-T', 'Ink staining']

    # 分类变量（独热编码）
    onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    # 连续变量
    numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

    # 验证所有特征都在数据中
    all_features = ordinal_features + onehot_features + numeric_features
    missing_features = [f for f in all_features if f not in X.columns]
    if missing_features:
        print(f"警告: 以下特征在数据中不存在: {missing_features}")
        # 从列表中移除不存在的特征
        ordinal_features = [f for f in ordinal_features if f in X.columns]
        onehot_features = [f for f in onehot_features if f in X.columns]
        numeric_features = [f for f in numeric_features if f in X.columns]

    return X, y_binary, ordinal_features, onehot_features, numeric_features


# 3. 构建预处理管道
def build_preprocessor(ordinal_features, onehot_features, numeric_features):
    """构建数据预处理管道"""

    # 数值特征：中位数填充
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 有序编码特征
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 独热编码特征
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('cat', onehot_transformer, onehot_features)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    return preprocessor


# 4. 构建完整模型管道
def build_model_pipeline(preprocessor):
    """构建包含SMOTE和随机森林的完整管道"""

    # 随机森林分类器
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=1  # 不使用多进程
    )

    # 使用imblearn的Pipeline支持SMOTE
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', rf)
    ])

    return pipeline


# 5. 超参数调优
def tune_hyperparameters(pipeline, X_train, y_train):
    """使用GridSearchCV进行超参数调优"""

    # 定义参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
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
    print("模型评估结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score): {f1:.4f}")
    print(f"AUC曲线下面积: {auc:.4f}")
    print("=" * 50)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc': auc
    }


# 7. 主函数
def main():
    """主执行函数"""

    # 数据文件路径
    file_path = 'CrAg_train.csv'

    # 加载数据
    print("正在加载数据...")
    df = load_data(file_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 数据预处理
    print("\n正在预处理数据...")
    X, y, ordinal_features, onehot_features, numeric_features = preprocess_data(df)

    print(f"有序编码特征: {ordinal_features}")
    print(f"独热编码特征: {onehot_features}")
    print(f"数值特征: {numeric_features}")
    print(f"目标分布: \n{y.value_counts()}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # 构建预处理器
    preprocessor = build_preprocessor(ordinal_features, onehot_features, numeric_features)

    # 构建模型管道
    pipeline = build_model_pipeline(preprocessor)

    # 超参数调优
    print("\n开始超参数调优...")
    best_model = tune_hyperparameters(pipeline, X_train, y_train)

    # 在测试集上评估最佳模型
    print("\n在测试集上评估最佳模型...")
    metrics = evaluate_model(best_model, X_test, y_test)

    # 可选：保存模型
    # import joblib
    # joblib.dump(best_model, 'best_model.pkl')
    # print("\n模型已保存为 'best_model.pkl'")

    return best_model, metrics


# 8. 脚本执行
if __name__ == "__main__":
    best_model, metrics = main()