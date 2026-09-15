import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 设置随机种子以确保结果可重复
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    # 1. 加载数据
    print("正在加载数据...")
    df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

    # 2. 分离特征和目标变量
    # 目标列是最后一列 CSF-T
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 将目标变量转换为二分类：是否 >= 20
    y_binary = (y >= 20).astype(int)

    print(f"数据集大小: {len(df)}")
    print(f"正样本 (>=20) 数量: {sum(y_binary == 1)}")
    print(f"负样本 (<20) 数量: {sum(y_binary == 0)}")
    print(f"不平衡比例: {sum(y_binary == 0) / sum(y_binary == 1):.2f}:1")

    # 3. 划分训练集和验证集 (4:1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_binary,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_binary  # 保持分类平衡
    )

    print(f"\n训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"训练集正样本比例: {sum(y_train) / len(y_train):.3f}")
    print(f"验证集正样本比例: {sum(y_val) / len(y_val):.3f}")

    # 4. 定义特征列
    # 分类变量：SER-T、Ink staining、Color、Transparency、SEX、DEPT、DIAGNOSIS
    # 其中 SER-T 和 Ink staining 使用序数编码
    # Color、Transparency、SEX、DEPT、DIAGNOSIS 使用独热编码

    # 连续变量：CL、GLU、Protein、RBC、AGE
    numerical_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

    # 序数编码特征 (有序分类变量)
    ordinal_features = ['SER-T', 'Ink staining']

    # 独热编码特征 (无序分类变量)
    onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

    # 验证所有特征都在数据集中
    all_features = numerical_features + ordinal_features + onehot_features
    missing_features = [f for f in all_features if f not in X_train.columns]
    if missing_features:
        print(f"警告: 以下特征在数据中不存在: {missing_features}")
        # 移除不存在的特征
        numerical_features = [f for f in numerical_features if f in X_train.columns]
        ordinal_features = [f for f in ordinal_features if f in X_train.columns]
        onehot_features = [f for f in onehot_features if f in X_train.columns]

    # 5. 构建预处理流水线
    # 预处理顺序：数值 → 独热 → 序数

    # 5.1 数值特征预处理：中位数填充
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 5.2 独热编码预处理
    onehot_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 5.3 序数编码预处理
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 5.4 组合所有预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('onehot', onehot_transformer, onehot_features),
            ('ordinal', ordinal_transformer, ordinal_features)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    # 6. 构建完整的流水线（含SMOTE，防止数据泄露）
    # 使用 imblearn 的 Pipeline 将 SMOTE 嵌入到交叉验证内部
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1  # 不使用多进程
        ))
    ])

    # 7. 定义超参数网格
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 8. 网格搜索（使用分层K折交叉验证，嵌入SMOTE防止数据泄露）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=1,  # 不使用多进程
        verbose=1
    )

    print("\n开始训练模型...")
    grid_search.fit(X_train, y_train)

    # 9. 输出最佳参数
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

    # 10. 在验证集上评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]  # 获取正类的概率

    # 计算评估指标
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # 计算AUC
    try:
        auc = roc_auc_score(y_val, y_pred_proba)
    except ValueError:
        # 如果验证集只有一个类别，无法计算AUC
        auc = float('nan')
        print("警告: 验证集只有一个类别，无法计算AUC")

    # 11. 输出评估结果
    print("\n" + "=" * 50)
    print("验证集评估结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score): {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("=" * 50)

    # 12. 额外信息：验证集上的分类报告
    from sklearn.metrics import classification_report
    print("\n详细分类报告:")
    print(classification_report(y_val, y_pred, target_names=['<20', '>=20']))

    # 13. 特征重要性（可选）
    # 由于流水线中包含了预处理和SMOTE，我们需要提取最终的分类器
    classifier = best_model.named_steps['classifier']

    # 获取特征名称（仅用于信息展示）
    # 获取独热编码后的特征名称
    try:
        # 获取独热编码特征名称
        onehot_encoder = best_model.named_steps['preprocessor'].named_transformers_['onehot'].named_steps['onehot']
        onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_features)

        # 序数编码特征名称
        ordinal_feature_names = ordinal_features

        # 数值特征名称
        numerical_feature_names = numerical_features

        # 所有特征名称
        feature_names = list(numerical_feature_names) + list(onehot_feature_names) + list(ordinal_feature_names)

        # 如果特征数量匹配，显示特征重要性
        if len(feature_names) == len(classifier.feature_importances_):
            print("\n特征重要性 (Top 10):")
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            print(importance_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"\n无法显示特征重要性: {e}")

    print("\n模型训练完成！")

    # 14. 保存模型（可选）
    # import joblib
    # joblib.dump(best_model, 'best_random_forest_model.pkl')
    # print("模型已保存为 best_random_forest_model.pkl")

    return best_model


if __name__ == "__main__":
    main()