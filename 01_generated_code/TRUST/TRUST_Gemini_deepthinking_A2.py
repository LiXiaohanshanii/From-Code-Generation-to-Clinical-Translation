import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. 读取数据
file_path = 'train_data.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 2. 构建二分类目标变量 (TRUST >= 16 为 1，否则为 0)
df['target'] = (df['TRUST'] >= 16).astype(int)

# 定义特征类型
cat_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
cat_ord_cols = ['TPPA']
num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

X = df[cat_ohe_cols + cat_ord_cols + num_cols]
y = df['target']

# 3. 划分训练集与测试集（按比例分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 构建特征预处理 Pipeline
# - 连续变量：中位数填充
# - 分类变量（SEX, DEPT, DIAGNOSIS）：独热编码
# - 分类变量（TPPA）：序数编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_ohe_cols),
        ('cat_ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_ord_cols)
    ]
)

# 对训练集和测试集应用特征转换（防止数据泄露）
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# 5. 使用 SMOTE 处理训练集的不平衡问题
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_prep, y_train)

# 6. 超参数调优（网格搜索，不使用多进程 n_jobs=1）
# 注：sklearn 中 '平衡' 对应的参数值为 'balanced'
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

rf_clf = RandomForestClassifier(random_state=42, n_jobs=1)

grid_search = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1  # 禁用多进程
)

# 拟合模型
grid_search.fit(X_train_resampled, y_train_resampled)

# 获取最佳模型
best_rf = grid_search.best_estimator_

# 7. 模型评估（在未经过 SMOTE 的原始测试集上进行）
y_pred = best_rf.predict(X_test_prep)
y_prob = best_rf.predict_proba(X_test_prep)[:, 1]

# 计算评估指标
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# 输出评估结果
print("=================== 最优超参数 ===================")
print(grid_search.best_params_)
print("\n=================== 模型评估结果 ===================")
print(f"准确率 (Accuracy)             : {acc:.4f}")
print(f"召回率 (Recall)               : {rec:.4f}")
print(f"精确率 (Precision)            : {prec:.4f}")
print(f"F1 分数 (F1-score)            : {f1:.4f}")
print(f"AUC 曲线下面积 (ROC-AUC Area) : {auc:.4f}")