import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# ===================== 1. 读取数据 =====================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ===================== 2. 分离特征与目标 =====================
# 目标列 CSF-T 是最后一列
target_col = df.columns[-1]
X = df.iloc[:, :-1]
y = df[target_col]

# ===================== 3. 构建二分类目标 =====================
# 预测 CSF-T 是否 >= 20
y_binary = (y >= 20).astype(int)

# ===================== 4. 定义变量类型 =====================
categorical_onehot = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal = ['SER-T', 'Ink staining']
continuous_vars = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# ===================== 5. 预处理管道 =====================

# 5.1 连续变量中位数填充
continuous_imputer = SimpleImputer(strategy='median')

# 5.2 序数编码
ordinal_encoder = OrdinalEncoder()

# 5.3 独热编码
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# ===================== 6. 手动构建预处理后的特征矩阵 =====================

# 6.1 处理连续变量
X_cont = pd.DataFrame(
    continuous_imputer.fit_transform(X[continuous_vars]),
    columns=continuous_vars
)

# 6.2 处理序数编码变量
X_ord = pd.DataFrame(
    ordinal_encoder.fit_transform(X[categorical_ordinal]),
    columns=categorical_ordinal
)

# 6.3 处理独热编码变量
X_ohe = pd.DataFrame(
    onehot_encoder.fit_transform(X[categorical_onehot]),
    columns=onehot_encoder.get_feature_names_out(categorical_onehot)
)

# 6.4 合并所有特征
X_processed = pd.concat([X_cont, X_ord, X_ohe], axis=1)

# ===================== 7. 划分训练集和测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ===================== 8. SMOTE 处理类别不平衡 =====================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"SMOTE前训练集类别分布: {np.bincount(y_train)}")
print(f"SMOTE后训练集类别分布: {np.bincount(y_train_smote)}")

# ===================== 9. 随机森林超参数调优 =====================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42, n_jobs=1)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    verbose=1,
    n_jobs=1
)

grid_search.fit(X_train_smote, y_train_smote)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ===================== 10. 使用最佳模型评估 =====================
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# ===================== 11. 计算评估指标 =====================
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 50)
print("模型评估结果（测试集）")
print("=" * 50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("=" * 50)