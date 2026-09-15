import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 忽略一些不必要的警告，保持控制台整洁
warnings.filterwarnings('ignore')


def main():
    # ==========================================
    # 1. 数据加载与目标变量处理
    # ==========================================
    print("正在加载数据...")
    # 读取数据，指定utf-8编码
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 目标列二值化：TRUST >= 16 标记为 1 (阳性/高滴度)，否则为 0
    # 注意：如果TRUST列包含字符串如"1:16"，需先进行清洗，这里假设已经是数值型(1, 2, 4, 8, 16...)
    df['TRUST_binary'] = (df['TRUST'] >= 16).astype(int)

    # 分离特征 (X) 和 目标变量 (y)
    X = df.drop(columns=['TRUST', 'TRUST_binary'])
    y = df['TRUST_binary']

    # ==========================================
    # 2. 划分训练集与测试集
    # ==========================================
    # 划分数据集，stratify=y 保证训练集和测试集中的正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")
    print(f"训练集正样本(>=16)比例: {y_train.mean():.2%}")

    # ==========================================
    # 3. 构建数据预处理管道 (ColumnTransformer)
    # ==========================================
    # 定义特征列 (修正了原提示中的拼写错误 DIAGONSIS -> DIAGNOSIS)
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ord_cols = ['TPPA']

    # 连续变量处理：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量处理1：独热编码 (One-Hot)
    # sparse_output=False 确保输出为密集矩阵，兼容后续的SMOTE
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量处理2：序数编码 (Ordinal)
    # 假设TPPA数据本身已具备顺序(如阴性, 1:80, 1:160)，若为纯文本需确保传入时已排序
    ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat_onehot', onehot_transformer, cat_onehot_cols),
            ('cat_ord', ord_transformer, cat_ord_cols)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    # ==========================================
    # 4. 构建包含 SMOTE 和 模型的完整 Pipeline
    # ==========================================
    # 使用 imblearn 的 Pipeline，确保 SMOTE 只在训练集的交叉验证折内执行，防止数据泄露
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    # 参数网格，注意需要加上 'classifier__' 前缀以指定 Pipeline 中的步骤
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("开始进行超参数网格搜索（不使用多进程）...")
    # n_jobs=1 明确禁止多进程，避免在PyCharm或Windows环境下出现子进程报错
    # scoring='roc_auc' 作为调优的评价指标，更适合不平衡数据
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"交叉验证最佳 AUC: {grid_search.best_score_:.4f}")

    # ==========================================
    # 6. 模型评估
    # ==========================================
    print("\n正在测试集上评估模型...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("-" * 30)
    print("【测试集评估指标】")
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}  <- (临床最关注：漏诊率)")
    print(f"精确率 (Precision) : {prec:.4f}  <- (临床关注：误诊率)")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积      : {auc:.4f}")
    print("-" * 30)


if __name__ == '__main__':
    main()
