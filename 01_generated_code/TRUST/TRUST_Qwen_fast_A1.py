import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ===================== 1. 读取数据 =====================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# ===================== 2. 构建目标变量（二分类：TRUST >= 16） =====================
y = (df['TRUST'] >= 16).astype(int)
X = df.drop(columns=['TRUST'])

# ===================== 3. 定义特征分组 =====================
categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal_cols = ['TPPA']
continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

# ===================== 4. 构建预处理管道 =====================
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_onehot_cols),
        ('ordinal', OrdinalEncoder(), categorical_ordinal_cols),
        ('median_impute', SimpleImputer(strategy='median'), continuous_cols)
    ],
    remainder='drop'
)

# ===================== 5. 划分训练集与测试集 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 6. 对训练集进行SMOTE过采样 =====================
# 注意：SMOTE需要在预处理之后、模型拟合之前执行，因此使用imblearn的Pipeline
# 先对训练集做预处理变换，再SMOTE，然后用处理后的数据做GridSearchCV
# 为避免数据泄露，这里手动分步处理训练集预处理+SMOTE，测试集仅预处理

# 拟合并变换训练集
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# SMOTE仅作用于训练集
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# ===================== 7. 超参数调优（随机森林） =====================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1,          # 不使用多进程
    verbose=0
)

grid_search.fit(X_train_resampled, y_train_resampled)

best_model = grid_search.best_estimator_
print(f"最佳超参数: {grid_search.best_params_}")
print(f"交叉验证最佳F1: {grid_search.best_score_:.4f}")

# ===================== 8. 在测试集上评估 =====================
y_pred = best_model.predict(X_test_processed)
y_prob = best_model.predict_proba(X_test_processed)[:, 1]

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n===== 测试集评估结果 =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
