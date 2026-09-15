import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # ==========================================
    # 1. 数据加载与预处理设置
    # ==========================================
    file_path = 'train_data.csv'

    # 读取数据（UTF-8编码）
    df = pd.read_csv(file_path, encoding='utf-8')

    # 提取特征列与目标列
    # 目标列 TRUST 是否 >= 16 转换为二分类任务 (1: >=16, 0: <16)
    target_col = 'TRUST'
    X = df.drop(columns=[target_col])
    y = (df[target_col] >= 16).astype(int)

    # 定义特征类型
    ohe_features = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码特征
    ord_features = ['TPPA']  # 序数编码特征
    num_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

    # ==========================================
    # 2. 构建特征转换器 (ColumnTransformer)
    # ==========================================
    # 连续变量：中位数填充
    # 分类变量（独热编码）：Sex, Dept, Diagnosis
    # 分类变量（序数编码）：TPPA
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_features),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_features),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ord_features)
        ],
        remainder='drop'
    )

    # ==========================================
    # 3. 划分训练集与测试集
    # ==========================================
    # 采用分层抽样 (stratify=y) 保证训练集和测试集中的正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 4. 构建包含 SMOTE 和随机森林的 Pipeline
    # ==========================================
    # 使用 imblearn.pipeline 确保 SMOTE 仅在交叉验证的训练折中执行，防止数据泄漏
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
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

    # 禁用多进程 (n_jobs=1)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    print("正在进行模型训练与超参数调优，请稍候...")
    grid_search.fit(X_train, y_train)

    # 最佳模型获取
    best_model = grid_search.best_estimator_
    print(f"\n最佳参数组合: {grid_search.best_params_}")

    # ==========================================
    # 6. 模型评估
    # ==========================================
    # 在测试集上进行预测
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 计算各项评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    # 打印结果
    print("\n" + "=" * 40)
    print("      随机森林分类模型评估结果 (TRUST >= 16)")
    print("=" * 40)
    print(f"准确率 (Accuracy):   {acc:.4f}")
    print(f"召回率 (Recall):     {rec:.4f}")
    print(f"精确率 (Precision):  {prec:.4f}")
    print(f"F1分数 (F1-Score):   {f1:.4f}")
    print(f"ROC-AUC (Area Under ROC): {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    main()