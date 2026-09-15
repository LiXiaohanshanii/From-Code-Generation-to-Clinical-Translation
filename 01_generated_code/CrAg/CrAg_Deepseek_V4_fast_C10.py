import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 设置随机种子以确保结果可重复
RANDOM_STATE = 42

# ============ 1. 数据加载 ============
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ============ 2. 分离特征和目标 ============
# 目标列是最后一列 'CSF-T'
target_col = 'CSF-T'
X = df.drop(columns=[target_col])
y = df[target_col]

# ============ 3. 将目标转换为二分类（>=20） ============
y_binary = (y >= 20).astype(int)

# ============ 4. 划分训练集和验证集（4:1） ============
X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
)

# ============ 5. 定义特征列 ============
# 分类变量（需要进行独热编码）
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

# 分类变量（需要进行序数编码）
ordinal_cols = ['SER-T', 'Ink staining']

# 连续变量（需要中位数填充）
numeric_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 验证所有特征列都在数据中
all_cols = onehot_cols + ordinal_cols + numeric_cols
assert set(all_cols) == set(X.columns), "特征列定义与数据不匹配"

# ============ 6. 创建预处理管道 ============
# 数值特征预处理：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 独热编码预处理（处理未知类别）
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 序数编码预处理
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合预处理器 - 按顺序：数值 → 独热 → 序数
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('onehot', onehot_transformer, onehot_cols),
        ('ordinal', ordinal_transformer, ordinal_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ============ 7. 构建包含SMOTE的完整Pipeline（防止数据泄露） ============
# 使用imblearn的Pipeline将SMOTE嵌入到交叉验证内部
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
])

# ============ 8. 定义超参数网格 ============
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# ============ 9. 网格搜索（使用f1作为评分指标） ============
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,  # 5折交叉验证
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 训练模型（SMOTE会在每次交叉验证折内自动应用，防止数据泄露）
grid_search.fit(X_train, y_train)

# ============ 10. 输出最佳参数 ============
print("=" * 60)
print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证F1分数:", grid_search.best_score_)
print("=" * 60)

# ============ 11. 在验证集上评估 ============
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]  # 获取正类概率

# 计算评估指标
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

# ============ 12. 打印评估结果 ============
print("验证集评估结果:")
print("-" * 40)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:               {auc:.4f}")
print("=" * 60)

# ============ 13. 可选：查看特征重要性 ============
# 获取预处理后的特征名称（用于理解模型）
try:
    # 获取列转换器中的特征名称
    preprocessor_fitted = best_model.named_steps['preprocessor']

    # 获取数值特征名称
    num_features = numeric_cols

    # 获取独热编码特征名称
    onehot_encoder = preprocessor_fitted.named_transformers_['onehot'].named_steps['onehot']
    onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_cols)

    # 获取序数编码特征名称
    ord_features = ordinal_cols

    # 组合所有特征名称
    all_feature_names = list(num_features) + list(onehot_feature_names) + list(ord_features)

    # 获取特征重要性
    feature_importance = best_model.named_steps['classifier'].feature_importances_

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\n特征重要性 (Top 10):")
    print("-" * 40)
    print(importance_df.head(10).to_string(index=False))
except Exception as e:
    print(f"\n无法获取特征重要性详情: {e}")

print("=" * 60)