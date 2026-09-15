import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. 读取数据
data = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 2. 分离特征和目标变量
# 目标列是CSF-T，为最后一列
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 3. 目标变量二分类处理（≥20为1，否则为0）
y_binary = (y >= 20).astype(int)

# 4. 定义特征类型
# 分类变量（需要进行编码）
categorical_cols_ordinal = ['SER-T', 'Ink staining']  # 序数编码
categorical_cols_onehot = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码

# 连续变量
numeric_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 检查数据中是否包含所有列
print("数据列名:", X.columns.tolist())

# 5. 数据预处理
# 5.1 为连续变量创建中位数填充处理器
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 5.2 为序数编码变量创建处理器（保留原始值，OrdinalEncoder需要指定类别顺序）
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 5.3 为独热编码变量创建处理器
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 5.4 组合所有预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('ord', ordinal_transformer, categorical_cols_ordinal),
        ('onehot', onehot_transformer, categorical_cols_onehot)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 7. 构建完整的Pipeline（包含SMOTE和随机森林）
# 注意：SMOTE需要在预处理之后、模型训练之前应用
# 使用imblearn的Pipeline以便正确处理
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 8. 定义要搜索的超参数
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 9. 创建GridSearchCV对象（不使用多进程，n_jobs=1）
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 10. 训练模型
print("开始训练模型...")
grid_search.fit(X_train, y_train)
print("训练完成！")

# 11. 输出最佳参数
print("\n最佳参数:", grid_search.best_params_)
print("最佳交叉验证F1分数: {:.4f}".format(grid_search.best_score_))

# 12. 在测试集上进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 对于AUC，需要预测概率
if hasattr(best_model, 'predict_proba'):
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
else:
    # 如果模型不支持predict_proba，则使用决策函数
    y_pred_proba = best_model.decision_function(X_test)

# 13. 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 14. 输出评估结果
print("\n=== 模型评估指标 ===")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC曲线下面积: {auc:.4f}")

# 15. 额外信息：各类别样本数量
print("\n=== 类别分布 ===")
print("训练集正负样本分布:")
print(pd.Series(y_train).value_counts())
print("\n测试集正负样本分布:")
print(pd.Series(y_test).value_counts())

# 16. 保存模型（可选）
import joblib

joblib.dump(best_model, 'best_random_forest_model.pkl')
print("\n模型已保存为 'best_random_forest_model.pkl'")

# 17. 查看特征重要性（可选）
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    # 获取特征名称（但经过preprocessor后，特征名称可能难以对应，这里仅做示例）
    print("\n特征重要性（前10个特征）:")
    importances = best_model.named_steps['classifier'].feature_importances_
    # 获取特征名称（简化处理）
    feature_names = []
    # 获取数值特征名称
    for col in numeric_cols:
        feature_names.append(col)
    # 获取序数编码特征名称
    for col in categorical_cols_ordinal:
        feature_names.append(col)
    # 获取独热编码特征名称（从预处理器中获取）
    onehot_feature_names = best_model.named_steps['preprocessor'].named_transformers_['onehot'].get_feature_names_out(
        categorical_cols_onehot)
    feature_names.extend(onehot_feature_names)

    # 创建特征重要性DataFrame并排序
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)
    print(feature_importance_df.head(10))