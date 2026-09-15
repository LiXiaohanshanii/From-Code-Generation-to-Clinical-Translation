# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path):
    """
    加载并准备数据
    """
    # 读取数据
    df = pd.read_csv(file_path, encoding='utf-8')

    # 确认目标列是最后一列
    target_col = df.columns[-1]

    # 分离特征和目标
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 将目标转换为二分类（是否≥20）
    y_binary = (y >= 20).astype(int)

    print(f"数据集形状: {df.shape}")
    print(f"特征列数: {X.shape[1]}")
    print(f"目标列: {target_col}")
    print(f"目标分布（≥20为1）: \n{y_binary.value_counts()}")
    print(f"目标分布比例: \n{y_binary.value_counts(normalize=True)}")

    return X, y_binary


def create_preprocessing_pipeline():
    """
    创建预处理管道
    """
    # 定义特征分类
    # 序数编码特征（OrdinalEncoder）
    ordinal_features = ['SER-T', 'Ink staining']

    # 独热编码特征（OneHotEncoder）
    onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    # 连续变量（需要中位数填充）
    continuous_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

    # 注意：如果数据集中的列名有拼写错误，在这里进行修正
    # 例如，如果实际列名是'Color'但数据中是'Colour'，需要相应调整

    # 创建预处理步骤
    # 1. 连续变量的预处理：中位数填充
    continuous_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 2. 序数编码
    ordinal_transformer = Pipeline([
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 3. 独热编码
    onehot_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 组合所有预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_transformer, continuous_features),
            ('ordinal', ordinal_transformer, ordinal_features),
            ('onehot', onehot_transformer, onehot_features)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    return preprocessor


def build_model_pipeline(preprocessor):
    """
    构建包含预处理和SMOTE的模型管道
    """
    # 创建包含SMOTE的完整管道
    # 注意：使用imblearn的Pipeline来支持SMOTE
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            random_state=42,
            n_jobs=1  # 禁用多进程
        ))
    ])

    return pipeline


def perform_grid_search(pipeline, X_train, y_train):
    """
    执行网格搜索进行超参数调优
    """
    # 定义参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 创建网格搜索对象
    # 注意：设置 n_jobs=1 禁用多进程
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=1,  # 禁用多进程
        verbose=1
    )

    print("开始网格搜索...")
    grid_search.fit(X_train, y_train)
    print("网格搜索完成！")

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

    return grid_search


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
    print("\n" + "=" * 50)
    print("模型评估结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score):  {f1:.4f}")
    print(f"AUC曲线下面积:      {auc:.4f}")
    print("=" * 50)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc': auc
    }


def main():
    """
    主函数
    """
    # 文件路径
    file_path = 'CrAg_train.csv'

    print("=" * 60)
    print("随机森林分类模型 - 预测CSF-T是否≥20")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    X, y = load_and_prepare_data(file_path)

    # 2. 划分训练集和测试集
    print("\n[2] 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")

    # 3. 创建预处理管道
    print("\n[3] 创建预处理管道...")
    preprocessor = create_preprocessing_pipeline()

    # 4. 构建完整模型管道
    print("\n[4] 构建模型管道（含SMOTE）...")
    pipeline = build_model_pipeline(preprocessor)

    # 5. 网格搜索调优
    print("\n[5] 超参数调优...")
    grid_search = perform_grid_search(pipeline, X_train, y_train)

    # 6. 在测试集上评估最佳模型
    print("\n[6] 在测试集上评估最佳模型...")
    best_model = grid_search.best_estimator_
    results = evaluate_model(best_model, X_test, y_test)

    print("\n模型构建和评估完成！")

    return best_model, results


if __name__ == "__main__":
    # 运行主程序
    model, metrics = main()