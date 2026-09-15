import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 加载数据 =====================
print("=" * 60)
print("正在加载数据...")
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')
print(f"数据集形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# ===================== 2. 分离特征与目标 =====================
# 目标列是最后一列 CSF-T
X = df.iloc[:, :-1]  # 所有特征列
y = df.iloc[:, -1]   # 最后一列 CSF-T

# 将目标转换为二分类：>=20 为 1，<20 为 0
y_binary = (y >= 20).astype(int)
print(f"\n目标列原始值分布:\n{y.value_counts().sort_index()}")
print(f"二分类目标分布:\n{y_binary.value_counts()}")

# ===================== 3. 定义变量类型 =====================
# 连续变量
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 独热编码变量
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

# 序数编码变量
ordinal_cols = ['SER-T', 'Ink staining']

# 验证所有列都被覆盖
all_feature_cols = continuous_cols + onehot_cols + ordinal_cols
print(f"\n特征列总数: {len(all_feature_cols)}")
print(f"连续变量: {continuous_cols}")
print(f"独热编码变量: {onehot_cols}")
print(f"序数编码变量: {ordinal_cols}")

# ===================== 4. 构建预处理管道 =====================
# 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 独热编码变量
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 序数编码变量
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 组合所有预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, continuous_cols),
        ('onehot', onehot_transformer, onehot_cols),
        ('ordinal', ordinal_transformer, ordinal_cols)
    ],
    remainder='passthrough'  # 保留其他未指定的列（如有）
)

# ===================== 5. 构建完整模型管道（含SMOTE） =====================
print("\n" + "=" * 60)
print("正在构建随机森林模型管道（含SMOTE）...")

# 随机森林分类器
rf_classifier = RandomForestClassifier(
    random_state=42,
    n_jobs=1  # 不使用多进程
)

# 完整管道：预处理 -> SMOTE -> 随机森林
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_classifier)
])

# ===================== 6. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"训练集正类比例: {y_train.mean():.4f}")
print(f"测试集正类比例: {y_test.mean():.4f}")

# ===================== 7. 超参数调优（GridSearchCV） =====================
print("\n" + "=" * 60)
print("正在进行超参数调优...")

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1,
    refit=True
)

grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ===================== 8. 模型评估 =====================
print("\n" + "=" * 60)
print("正在评估模型...")

best_model = grid_search.best_estimator_

# 在测试集上预测
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("模型评估结果（测试集）:")
print("=" * 60)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("=" * 60)

# ===================== 9. 额外信息 =====================
print("\n混淆矩阵:")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0][0]}, FP={cm[0][1]}")
print(f"  FN={cm[1][0]}, TP={cm[1][1]}")

print("\n模型构建完成！")