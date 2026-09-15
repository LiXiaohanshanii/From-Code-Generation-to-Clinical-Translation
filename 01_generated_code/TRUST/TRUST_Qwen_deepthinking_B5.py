import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
# 注意：必须使用 imblearn 的 Pipeline，以确保 SMOTE 只在交叉验证的训练折中应用
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # ==========================================
    # 1. 数据加载与目标变量转换
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 目标变量处理：TRUST滴度 >= 16 标记为 1 (阳性/高滴度)，否则为 0
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # ==========================================
    # 2. 数据集划分
    # ==========================================
    # 使用分层抽样(stratify)保证训练集和测试集中正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 数据预处理管道构建
    # ==========================================
    # 定义特征列
    continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    categorical_ohe_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ord_features = ['TPPA']

    # 构建 ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            # 连续变量：中位数填充
            ('num', SimpleImputer(strategy='median'), continuous_features),
            # 分类变量(独热)：One-Hot 编码，handle_unknown='ignore' 防止测试集出现新类别报错
            ('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_ohe_features),
            # 分类变量(序数)：Ordinal 编码，处理未知类别映射为 -1
            ('cat_ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_ord_features)
        ]
    )

    # ==========================================
    # 4. 构建包含 SMOTE 和 随机森林 的 Pipeline
    # ==========================================
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),  # 处理分类不平衡
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("开始进行超参数网格搜索调优（不使用多进程）...")
    # n_jobs=1 确保不使用多进程；scoring='roc_auc' 在医学二分类中通常是最稳健的调优指标
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数组合: {grid_search.best_params_}")
    print(f"交叉验证最佳 AUC 得分: {grid_search.best_score_:.4f}\n")

    # ==========================================
    # 6. 模型评估
    # ==========================================
    best_model = grid_search.best_estimator_

    # 在测试集上进行预测
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率用于计算AUC

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估结果
    print("=" * 40)
    print("测试集模型评估结果：")
    print("=" * 40)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}")
    print(f"精确率 (Precision) : {prec:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积 (AUC): {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
