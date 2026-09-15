import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # ==========================================
    # 1. 数据加载与目标变量转换
    # ==========================================
    print("正在加载数据...")
    # 读取数据，指定utf-8编码
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 目标列转换：TRUST >= 16 标记为 1 (阳性/高风险)，否则为 0
    # 临床意义：TRUST滴度>=1:16通常提示活动期感染或需要积极治疗
    y = (df['TRUST'] >= 16).astype(int)
    X = df.drop(columns=['TRUST'])

    print(f"数据集形状: {X.shape}")
    print(f"目标变量分布:\n{y.value_counts(normalize=True)}")

    # ==========================================
    # 2. 划分训练集与测试集
    # ==========================================
    # 使用 stratify=y 确保训练集和测试集中正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 数据预处理管道构建
    # ==========================================
    # 定义特征列表
    numeric_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    cat_onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ordinal_features = ['TPPA']

    # 连续变量：中位数填充
    numeric_transformer = SimpleImputer(strategy='median')

    # 分类变量（独热编码）：注意 sparse_output=False 以兼容 SMOTE
    # handle_unknown='ignore' 防止测试集中出现训练集未见过的类别报错
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量（序数编码）：TPPA
    # handle_unknown='use_encoded_value' 防止测试集出现未知类别报错
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat_onehot', onehot_transformer, cat_onehot_features),
            ('cat_ordinal', ordinal_transformer, cat_ordinal_features)
        ],
        remainder='drop'  # 丢弃未明确指定的列
    )

    # ==========================================
    # 4. 构建包含 SMOTE 的完整 Pipeline
    # ==========================================
    # 注意：必须使用 imblearn.pipeline.Pipeline，以确保 SMOTE 仅在训练集的 CV 折叠中应用，防止数据泄露
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))  # n_jobs=1 禁用多进程
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

    print("\n开始进行超参数网格搜索 (不使用多进程，请耐心等待)...")
    # 评估指标选择 roc_auc，因为它对不平衡数据更鲁棒
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,  # 明确要求不使用多进程
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数组合: {grid_search.best_params_}")
    print(f"交叉验证最佳 ROC-AUC 得分: {grid_search.best_score_:.4f}")

    # ==========================================
    # 6. 模型评估
    # ==========================================
    best_model = grid_search.best_estimator_

    # 在测试集上进行预测 (imblearn Pipeline 在 predict 时会自动跳过 SMOTE)
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算各项评估指标
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估结果
    print("\n" + "=" * 40)
    print("测试集模型评估指标 (TRUST >= 16 预测)")
    print("=" * 40)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {recall:.4f}  <- 临床最关注，尽量不漏诊高滴度患者")
    print(f"精确率 (Precision) : {precision:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积      : {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
