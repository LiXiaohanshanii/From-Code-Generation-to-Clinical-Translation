import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ============================================================
# 2. 分离特征与目标
# ============================================================
# 目标列是最后一列 CSF-T
target_col = df.columns[-1]
X = df.iloc[:, :-1]
y = df[target_col]

# 构建二分类标签：>=20 为 1，<20 为 0
y_binary = (y >= 20).astype(int)

# ============================================================
# 3. 定义变量类型
# ============================================================
# 分类变量（独热编码）
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
# 分类变量（序数编码）
ordinal_cols = ['SER-T', 'Ink staining']
# 连续变量
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 确保所有列都在X中
all_feature_cols = onehot_cols + ordinal_cols + continuous_cols
X = X[all_feature_cols].copy()

# ============================================================
# 4. 预处理管道
# ============================================================
# 连续变量：中位数填充
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 序数编码
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 独热编码
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 组合列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, continuous_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('cat', onehot_transformer, onehot_cols)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# ============================================================
# 5. 划分训练集和测试集
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ============================================================
# 6. 构建完整Pipeline（含SMOTE + 随机森林）
# ============================================================
# 注意：SMOTE需要放在Pipeline之外，因为它需要在预处理之后操作
# 所以先对训练数据进行预处理
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# SMOTE过采样
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# ============================================================
# 7. 随机森林模型 + GridSearchCV 超参数调优
# ============================================================
rf = RandomForestClassifier(random_state=42, n_jobs=1)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,          # 不使用多进程
    verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

# 最佳模型
best_rf = grid_search.best_estimator_
print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ============================================================
# 8. 模型评估
# ============================================================
# 预测
y_pred = best_rf.predict(X_test_processed)
y_prob = best_rf.predict_proba(X_test_processed)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*50)
print("模型评估结果（测试集）")
print("="*50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):     {recall:.4f}")
print(f"精确率 (Precision):  {precision:.4f}")
print(f"F1分数 (F1-score):   {f1:.4f}")
print(f"AUC:                 {auc:.4f}")
print("="*50)