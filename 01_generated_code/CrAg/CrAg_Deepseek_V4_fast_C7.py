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
import warnings
warnings.filterwarnings('ignore')

# 1. 读取数据
df = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 2. 分离特征和目标变量
# 目标列是最后一列 CSF-T
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. 将目标变量转换为二分类：是否 >= 20
y_binary = (y >= 20).astype(int)

# 4. 划分训练集和验证集（4:1）
X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 5. 定义特征列
# 分类变量（独热编码）：Color, Transparency, SEX, DEPT, DIAGNOSIS
onehot_features = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']

# 分类变量（序数编码）：SER-T, Ink staining
ordinal_features = ['SER-T', 'Ink staining']

# 连续变量（中位数填充）：CL, GLU, Protein, RBC, AGE
numeric_features = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 6. 构建预处理步骤
# 6.1 数值变量处理：中位数填充
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 6.2 独热编码变量处理
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 6.3 序数编码变量处理
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 7. 组合预处理（按顺序：数值 → 独热 → 序数）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('onehot', onehot_transformer, onehot_features),
        ('ord', ordinal_transformer, ordinal_features)
    ],
    remainder='drop'  # 丢弃未指定的列
)

# 8. 构建包含SMOTE的完整Pipeline（防止数据泄露）
# 使用imblearn的Pipeline确保SMOTE在交叉验证内部进行
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 9. 定义超参数网格
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# 10. 网格搜索（使用f1作为评分指标，不使用多进程）
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 11. 训练模型
print("开始训练模型...")
grid_search.fit(X_train, y_train)
print("训练完成！")

# 12. 获取最佳模型
best_model = grid_search.best_estimator_
print(f"\n最佳参数组合: {grid_search.best_params_}")

# 13. 在验证集上进行预测
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

# 14. 计算评估指标
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

# 15. 输出评估结果
print("\n" + "="*50)
print("模型评估结果（验证集）")
print("="*50)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC曲线下面积:      {auc:.4f}")
print("="*50)

# 16. 可选：输出分类报告
from sklearn.metrics import classification_report
print("\n详细分类报告:")
print(classification_report(y_val, y_pred, target_names=['<20', '>=20']))