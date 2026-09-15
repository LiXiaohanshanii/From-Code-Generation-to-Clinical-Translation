import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
# 注意：必须使用 imblearn 的 Pipeline，因为 SMOTE 会改变样本数量，sklearn 原生 Pipeline 不支持
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # ==========================================
    # 1. 加载数据与目标变量处理
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 确保 TRUST 列为数值类型，无法转换的脏数据转为 NaN 并剔除
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')
    df = df.dropna(subset=['TRUST'])

    # 目标列二分类处理：>=16 为 1，否则为 0
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    # ==========================================
    # 2. 划分训练集与测试集
    # ==========================================
    # 使用 stratify=y 确保训练集和测试集中的正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 定义特征列与数据预处理管道
    # ==========================================
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 修正了提示词中的拼写错误 DIAGONSIS
    cat_ordinal_cols = ['TPPA']

    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量 (独热编码)：处理未知类别
    cat_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量 (序数编码)：处理未知类别，将其映射为 -1
    cat_ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat_onehot', cat_onehot_transformer, cat_onehot_cols),
            ('cat_ordinal', cat_ordinal_transformer, cat_ordinal_cols)
        ]
    )

    # ==========================================
    # 4. 构建完整 Pipeline (包含预处理、SMOTE和模型)
    # ==========================================
    model_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),  # 在 Pipeline 内执行 SMOTE 可防止交叉验证时的数据泄露
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10],
        'clf__min_samples_split': [2],
        'clf__min_samples_leaf': [1],
        'clf__class_weight': ['balanced']
    }

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring='f1',  # 针对不平衡数据，使用 F1-score 作为调优的主要评估指标
        n_jobs=1,  # 明确指定不使用多进程
        verbose=1  # 打印调优进度
    )

    # ==========================================
    # 6. 模型训练
    # ==========================================
    print("\n开始进行超参数调优和模型训练...")
    grid_search.fit(X_train, y_train)

    print(f"\n最佳超参数组合: {grid_search.best_params_}")

    # ==========================================
    # 7. 模型评估
    # ==========================================
    best_model = grid_search.best_estimator_

    # 预测类别与预测概率
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 计算各项评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 打印评估结果
    print("\n" + "=" * 30)
    print("       模型评估结果       ")
    print("=" * 30)
    print(f"准确率 (Accuracy)  : {accuracy:.4f}")
    print(f"召回率 (Recall)    : {recall:.4f}")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积 (AUC): {auc:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
