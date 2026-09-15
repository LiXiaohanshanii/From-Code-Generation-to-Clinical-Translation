import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# ===================== 1. 数据加载与目标变量构建 =====================
df = pd.read_csv('train_data.csv', encoding='utf-8')

# 定义特征列
cat_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
cat_cols_ordinal = ['TPPA']
num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

feature_cols = cat_cols_onehot + cat_cols_ordinal + num_cols
X = df[feature_cols].copy()

# 构建二分类目标: TRUST >= 16 为阳性(1), 否则为阴性(0)
y = (df['TRUST'] >= 16).astype(int)

# ===================== 2. 构建预处理Pipeline =====================
# 分类变量独热编码（无缺失值，handle_unknown='ignore'防止测试集出现新类别报错）
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 连续变量中位数填充
num_transformer = SimpleImputer(strategy='median')

# TPPA序数编码（假设TPPA的有序类别为字符串形式的滴度描述，按实际数据中的唯一值排序）
# 注意：若TPPA原始值为数值型滴度如"1:1","1:2"等字符串，需确保categories顺序正确
# 此处使用默认自动推断排序，若业务有明确顺序请手动指定 categories=[['阴性','1:1','1:2',...]]
ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', onehot_transformer, cat_cols_onehot),
        ('ordinal', ordinal_transformer, cat_cols_ordinal),
        ('num', num_transformer, num_cols)
    ],
    remainder='drop'
)

# ===================== 3. 构建含SMOTE的完整建模Pipeline =====================
# 使用imblearn的Pipeline确保SMOTE仅在训练折内执行，避免数据泄露
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ===================== 4. 超参数调优（不使用多进程） =====================
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='f1',          # 不平衡数据以F1为主要优化指标
    n_jobs=1,              # 明确要求不使用多进程
    verbose=1,
    refit=True
)

print("开始超参数搜索...")
grid_search.fit(X, y)

print(f"\n最优参数: {grid_search.best_params_}")
print(f"最优交叉验证F1分数: {grid_search.best_score_:.4f}")

# ===================== 5. 最终模型评估（基于最优模型在全量数据上的表现） =====================
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)
y_prob = best_model.predict_proba(X)[:, 1]

acc = accuracy_score(y, y_pred)
rec = recall_score(y, y_pred)
prec = precision_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)

print("\n===== 最终模型评估结果 =====")
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC:                {auc:.4f}")
