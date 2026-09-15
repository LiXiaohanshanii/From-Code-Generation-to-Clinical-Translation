import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_selector as selector

# 1. 读取数据
data = pd.read_csv('CrAg_train.csv', encoding='utf-8')

# 2. 分离特征和目标变量（目标列CSF-T是最后一列）
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 3. 将目标变量转换为二分类（≥20为1，<20为0）
y_binary = (y >= 20).astype(int)

# 4. 划分训练集和验证集（4:1比例）
X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 5. 定义列分组
# 分类变量编码方式：
# - 独热编码：Color, Transparency, SEX, DEPT, DIAGNOSIS
# - 序数编码：SER-T, Ink staining
# 连续变量：CL, GLU, Protein, RBC, AGE

categorical_onehot_cols = ['Color', 'Transparency', 'SEX', 'DEPT', 'DIAGNOSIS']
categorical_ordinal_cols = ['SER-T', 'Ink staining']
numerical_cols = ['CL', 'GLU', 'Protein', 'RBC', 'AGE']

# 6. 构建预处理步骤（严格按照顺序：数值→独热→序数）

# 6.1 数值变量预处理：中位数填充
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# 6.2 独热编码变量预处理
onehot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 6.3 序数编码变量预处理
ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 6.4 组合所有预处理步骤（顺序：数值→独热→序数）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('onehot', onehot_transformer, categorical_onehot_cols),
        ('ordinal', ordinal_transformer, categorical_ordinal_cols)
    ],
    remainder='drop'  # 删除未指定的列
)

# 7. 构建包含SMOTE的完整Pipeline（防止数据泄露）
# 重要：SMOTE必须在训练集内部进行，使用imblearn的Pipeline确保在交叉验证时正确执行
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
])

# 8. 超参数调优（使用GridSearchCV）
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [1],
    'classifier__class_weight': ['balanced']
}

# GridSearchCV内部使用5折交叉验证，评分指标为f1
# 注意：由于使用了ImbPipeline，SMOTE会在每一折的训练集内部执行，不会泄漏到验证集
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,  # 不使用多进程
    verbose=1
)

# 9. 训练模型
print("开始训练模型...")
grid_search.fit(X_train, y_train)
print("训练完成！")

# 10. 获取最佳模型
best_model = grid_search.best_estimator_
print(f"\n最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 11. 在验证集上进行预测
y_pred = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]  # 获取正类概率用于AUC

# 12. 计算评估指标
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_proba)

# 13. 输出评估结果
print("\n" + "="*60)
print("模型在验证集上的评估结果:")
print("="*60)
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"F1分数 (F1-score):  {f1:.4f}")
print(f"AUC (ROC曲线下面积): {auc:.4f}")
print("="*60)

# 14. 可选：输出验证集的详细分类报告
from sklearn.metrics import classification_report
print("\n详细分类报告:")
print(classification_report(y_val, y_pred, target_names=['<20', '≥20']))

# 15. 验证数据预处理是否按正确顺序执行
# 可以通过查看preprocessor的transformers顺序来确认
print("\n数据预处理顺序确认:")
print("1. 数值变量 (中位数填充)")
print("2. 独热编码 (Color, Transparency, SEX, DEPT, DIAGNOSIS)")
print("3. 序数编码 (SER-T, Ink staining)")
print("\nSMOTE在预处理之后、分类器之前执行，且嵌入在交叉验证内部。")

# 16. 保存最佳模型（可选）
# import joblib
# joblib.dump(best_model, 'best_random_forest_model.pkl')
# print("\n模型已保存为 'best_random_forest_model.pkl'")