import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

# ==========================
# 1. 数据加载与目标变量构建
# ==========================
# 读取数据集
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 构建二分类目标变量: CSF-T >= 20 为 1, 否则为 0
# 注意: 确保CSF-T列被正确解析为数值类型
df['target'] = (pd.to_numeric(df['CSF-T'], errors='coerce') >= 20).astype(int)

# 定义特征列和目标列
feature_cols = ['SER-T', 'CL', 'GLU', 'Ink staining', 'Color',
                'Protein', 'Transparency', 'RBC', 'SEX', 'AGE',
                'DEPT', 'DIAGNOSIS']
target_col = 'target'

X = df[feature_cols]
y = df[target_col]

# ==========================
# 2. 定义预处理步骤
# ==========================
# 分类变量分组
# 序数编码变量 (根据输入数据描述修正)
ordinal_features = ['SER-T', 'Ink staining']
# 独热编码变量
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
# 连续变量
numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 构建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_features)
    ],
    remainder='drop'
)

# ==========================
# 3. 构建包含SMOTE的Pipeline
# ==========================
# 使用imblearn的Pipeline以确保SMOTE仅在训练集的交叉验证折叠内执行
# 防止数据泄露
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# ==========================
# 4. 超参数调优 (GridSearchCV)
# ==========================
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
    scoring='f1',           # 以F1作为评分指标
    cv=5,                   # 5折交叉验证
    n_jobs=1,               # 不使用多进程，符合任务要求
    verbose=0
)

print("开始模型训练与超参数调优...")
grid_search.fit(X, y)

print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证 F1-score: {grid_search.best_score_:.4f}\n")

# ==========================
# 5. 模型评估 (基于最佳模型的交叉验证预测)
# ==========================
# 获取最佳模型在交叉验证中的预测结果用于评估
# 注意: 这里使用cross_val_predict会更严谨，但为保持与调优一致，
# 我们重新用最佳参数拟合后通过交叉验证获取指标
from sklearn.model_selection import cross_val_predict

best_model = grid_search.best_estimator_
# 获取交叉验证预测概率和标签
y_pred_cv = cross_val_predict(best_model, X, y, cv=5, method='predict')
y_prob_cv = cross_val_predict(best_model, X, y, cv=5, method='predict_proba')[:, 1]

# 计算各项评估指标
acc = accuracy_score(y, y_pred_cv)
rec = recall_score(y, y_pred_cv)
prec = precision_score(y, y_pred_cv)
f1 = f1_score(y, y_pred_cv)
auc = roc_auc_score(y, y_prob_cv)

print("=" * 40)
print("   随机森林模型评估结果 (5-Fold CV)")
print("=" * 40)
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"召回率 (Recall):    {rec:.4f}")
print(f"精确率 (Precision): {prec:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC曲线下面积:      {auc:.4f}")
print("=" * 40)