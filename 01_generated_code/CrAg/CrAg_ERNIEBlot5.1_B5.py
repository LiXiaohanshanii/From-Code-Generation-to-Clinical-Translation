import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score,
                             roc_auc_score, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')

# ========================
# 1. 读取数据
# ========================
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# ========================
# 2. 定义列
# ========================
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color',
                'Protein', 'Transparency', 'RBC', 'SEX', 'AGE',
                'DEPT', 'DIAGNOSIS']
target_col = 'CSF-T'

X = df[feature_cols].copy()
y = df[target_col].copy()

# ========================
# 3. 构造二分类目标：CSF-T >= 20 ?
# ========================
y_binary = (y >= 20).astype(int)

# ========================
# 4. 分类变量与连续变量分组
# ========================
onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
ordinal_cols = ['SER-T', 'Ink staining']
continuous_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# ========================
# 5. 缺失值处理（仅连续变量，用中位数填充）
# ========================
cont_imputer = SimpleImputer(strategy='median')
X[continuous_cols] = cont_imputer.fit_transform(X[continuous_cols])

# ========================
# 6. 分类变量编码
# ========================
# 序数编码
ord_encoder = OrdinalEncoder()
X[ordinal_cols] = ord_encoder.fit_transform(X[ordinal_cols])

# 独热编码
ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_result = ohe_encoder.fit_transform(X[onehot_cols])
ohe_feature_names = ohe_encoder.get_feature_names_out(onehot_cols)

# 合并所有特征
X_continuous = X[continuous_cols].values
X_ordinal = X[ordinal_cols].values
X_ohe = ohe_result

X_final = np.hstack([X_continuous, X_ordinal, X_ohe])

# ========================
# 7. 划分训练集/测试集
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ========================
# 8. SMOTE 过采样（仅对训练集）
# ========================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"SMOTE前 训练集类别分布: {np.bincount(y_train)}")
print(f"SMOTE后 训练集类别分布: {np.bincount(y_train_res)}")

# ========================
# 9. 随机森林 + 网格搜索
# ========================
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
    verbose=0,
    n_jobs=1
)

grid_search.fit(X_train_res, y_train_res)

best_model = grid_search.best_estimator_
print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1: {grid_search.best_score_:.4f}")

# ========================
# 10. 模型评估（测试集）
# ========================
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 50)
print("模型评估结果（测试集）")
print("=" * 50)
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"精确率 (Precision): {pre:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
print("=" * 50)
print("\n分类报告:")
print(classification_report(y_test, y_pred,
                            target_names=['CSF-T < 20', 'CSF-T >= 20']))