import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
# 使用 imblearn 的 Pipeline 以支持在流水线中插入 SMOTE，防止测试集数据泄露
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # ==========================================
    # 1. 加载数据与目标变量转换
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 确保 TRUST 列为数值类型（处理可能存在的脏数据）
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')

    # 目标变量转换：预测 TRUST 是否 >= 16 (1表示>=16, 0表示<16)
    df['target'] = (df['TRUST'] >= 16).astype(int)

    # 定义特征列
    continuous_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    onehot_features = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_features = ['TPPA']

    # 划分特征 X 和 目标 y
    X = df[continuous_features + onehot_features + ordinal_features]
    y = df['target']

    # ==========================================
    # 2. 划分训练集与测试集
    # ==========================================
    print("正在划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 构建数据预处理 Pipeline
    # ==========================================
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量（独热编码）：处理未知类别，输出密集矩阵以兼容SMOTE
    cat_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量（序数编码）：处理未知类别
    cat_ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, continuous_features),
            ('cat_onehot', cat_onehot_transformer, onehot_features),
            ('cat_ordinal', cat_ordinal_transformer, ordinal_features)
        ]
    )

    # 构建包含预处理、SMOTE和分类器的完整 Pipeline
    # 注意：SMOTE 放在 Pipeline 中可以确保交叉验证时只在训练折上采样，避免数据泄露
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))  # n_jobs=1 禁用多进程
    ])

    # ==========================================
    # 4. 超参数调优 (GridSearchCV)
    # ==========================================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("开始进行超参数网格搜索（单进程运行，请耐心等待）...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring='f1',  # 针对不平衡数据，使用 F1-score 作为调优指标
        n_jobs=1,  # 严格限制不使用多进程
        verbose=1  # 打印搜索进度
    )

    grid_search.fit(X_train, y_train)

    print(f"\n最佳超参数组合: {grid_search.best_params_}")
    print(f"交叉验证最佳 F1-score: {grid_search.best_score_:.4f}")

    # ==========================================
    # 5. 模型评估
    # ==========================================
    best_model = grid_search.best_estimator_

    # 在测试集上进行预测
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率用于计算AUC

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 打印评估结果
    print("\n" + "=" * 30)
    print("测试集模型评估指标：")
    print("=" * 30)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}")
    print(f"精确率 (Precision) : {pre:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积 (AUC): {auc:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
