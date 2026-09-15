import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# 1. 加载数据
print("正在加载数据...")
data = pd.read_csv('train_data.csv', encoding='utf-8')

# 2. 分离特征和目标变量
X = data.iloc[:, :-1]  # 所有特征列
y = data.iloc[:, -1]  # 目标列 TRUST

# 3. 将目标变量转换为二分类（>=16 为1，<16为0）
y_binary = (y >= 16).astype(int)
print(f"目标变量分布：\n{y_binary.value_counts()}")

# 4. 定义特征类型
# 分类变量（无缺失值）
categorical_features = ['SEX', 'DEPT', 'DIAGNOSIS']
# TPPA使用序数编码
ordinal_features = ['TPPA']
# 连续变量（需要中位数填充）
numerical_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# 5. 创建预处理管道
# 5.1 数值特征处理器：中位数填充
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 5.2 分类特征处理器：独热编码
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# 5.3 序数特征处理器：序数编码
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

# 5.4 组合所有处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features)
    ])

# 6. 创建完整的管道（包含SMOTE和随机森林）
# 注意：SMOTE需要在预处理之后、模型训练之前进行
# 但为了在交叉验证中正确处理，我们使用imblearn的Pipeline

# 先创建一个不包含SMOTE的预处理+模型管道用于查看特征数量
# 实际上我们需要在预处理后应用SMOTE，所以分开步骤

# 6.1 单独预处理数据以确定特征数量（用于参数调优，但这里我们用Pipeline统一处理）
# 我们使用ImbPipeline将预处理、SMOTE和模型整合在一起

# 定义随机森林模型
rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=1  # 不使用多进程
)

# 创建完整的管道
# 注意：SMOTE应该在预处理之后应用，但在模型训练之前
# 使用imblearn的Pipeline可以确保在交叉验证中正确应用SMOTE
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

# 7. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 8. 定义超参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 9. 进行网格搜索
print("\n开始超参数调优...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

grid_search.fit(X_train, y_train)

# 10. 输出最佳参数
print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# 11. 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 12. 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# 13. 输出评估结果
print("\n" + "=" * 50)
print("模型评估结果")
print("=" * 50)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score): {f1:.4f}")
print(f"AUC值: {auc:.4f}")
print("=" * 50)

# 14. 额外信息：特征重要性（可选）
# 获取特征名称
try:
    # 获取预处理后的特征名称
    preprocessor_fitted = best_model.named_steps['preprocessor']

    # 获取数值特征名称
    num_features = numerical_features

    # 获取独热编码特征名称
    cat_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out(categorical_features)

    # 获取序数特征名称
    ord_features = ordinal_features

    # 组合所有特征名称
    feature_names = list(num_features) + list(cat_features) + list(ord_features)

    # 获取特征重要性
    classifier = best_model.named_steps['classifier']
    importances = classifier.feature_importances_

    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n特征重要性（前10个特征）:")
    print(feature_importance_df.head(10))

except Exception as e:
    print(f"\n无法获取特征重要性: {e}")

# 15. 保存模型（可选）
import joblib

joblib.dump(best_model, 'random_forest_model.pkl')
print("\n模型已保存为 'random_forest_model.pkl'")

print("\n程序执行完成！")