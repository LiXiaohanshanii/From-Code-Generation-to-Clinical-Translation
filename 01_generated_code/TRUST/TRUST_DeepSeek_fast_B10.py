import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子以保证结果可重复
RANDOM_STATE = 42


def load_data(file_path='train_data.csv'):
    """
    加载数据集
    """
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        print(f"数据集加载成功，共有 {data.shape[0]} 行，{data.shape[1]} 列")
        return data
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"加载数据时出错：{e}")
        return None


def create_preprocessor():
    """
    创建数据预处理管道
    """
    # 定义特征类型
    categorical_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']
    numeric_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 分类变量预处理
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # 序数编码预处理
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 连续变量预处理（中位数填充）
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 组合所有预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('ord', ordinal_transformer, ordinal_cols),
            ('num', numeric_transformer, numeric_cols)
        ]
    )

    return preprocessor


def create_model():
    """
    创建随机森林模型（配置为不使用多进程）
    """
    return RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1  # 不使用多进程
    )


def prepare_data(data):
    """
    准备数据：分离特征和目标变量，进行训练集和测试集划分
    """
    # 目标变量是最后一列（TRUST）
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 将目标变量转换为二分类（是否 >= 16）
    y_binary = (y >= 16).astype(int)

    # 划分训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
    )

    print(f"\n数据集划分结果：")
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    print(f"训练集中正例比例: {y_train.mean():.4f}")
    print(f"测试集中正例比例: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'auc': auc
    }


def main():
    """
    主函数：执行完整的模型构建和评估流程
    """
    print("=" * 60)
    print("随机森林分类模型 - 构建与评估")
    print("=" * 60)

    # 1. 加载数据
    data = load_data('train_data.csv')
    if data is None:
        return

    # 2. 数据探索
    print(f"\n数据基本信息：")
    print(f"特征列: {list(data.columns[:-1])}")
    print(f"目标列: {data.columns[-1]}")
    print(f"\n目标列值分布:")
    print(data.iloc[:, -1].value_counts().sort_index())

    # 3. 准备数据
    X_train, X_test, y_train, y_test = prepare_data(data)

    # 4. 创建预处理器和模型
    preprocessor = create_preprocessor()
    model = create_model()

    # 5. 创建包含SMOTE的完整管道
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', model)
    ])

    # 6. 定义超参数网格（注意参数名称要匹配管道中的步骤名）
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 7. 网格搜索（不使用多进程）
    print("\n开始超参数调优...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # 5折交叉验证
        scoring='f1',  # 使用F1分数作为优化指标
        n_jobs=1,  # 不使用多进程
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # 8. 输出最佳参数
    print(f"\n最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

    # 9. 在测试集上评估最佳模型
    best_model = grid_search.best_estimator_

    print("\n在测试集上评估模型...")
    metrics = evaluate_model(best_model, X_test, y_test)

    # 10. 输出评估结果
    print("\n" + "=" * 60)
    print("模型评估结果（测试集）")
    print("=" * 60)
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"F1分数 (F1-score):  {metrics['f1_score']:.4f}")
    print(f"AUC值 (AUC-ROC):    {metrics['auc']:.4f}")
    print("=" * 60)

    return best_model, metrics


if __name__ == "__main__":
    main()