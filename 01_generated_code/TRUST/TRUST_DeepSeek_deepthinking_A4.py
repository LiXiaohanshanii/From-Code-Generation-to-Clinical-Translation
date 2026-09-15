import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 设置随机种子保证可复现性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    # 1. 读取数据
    data = pd.read_csv('train_data.csv', encoding='utf-8')

    # 2. 特征与目标分离（目标为最后一列TRUST）
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]  # TRUST列

    # 将目标转换为二分类：是否 ≥16
    y_binary = (y >= 16).astype(int)

    # 3. 划分训练集和测试集（保持类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, stratify=y_binary, random_state=RANDOM_STATE
    )

    # 定义特征列名
    categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码
    categorical_ordinal_cols = ['TPPA']  # 序数编码
    numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 4. 预处理流水线：数值特征中位数填充 + 分类特征编码
    # 数值特征：中位数填充
    numerical_transformer = SimpleImputer(strategy='median')

    # 分类特征：独热编码（handle_unknown='ignore' 防止测试集出现新类别）
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类特征：序数编码（类别顺序按训练集出现顺序分配）
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('onehot', onehot_transformer, categorical_onehot_cols),
            ('ordinal', ordinal_transformer, categorical_ordinal_cols)
        ]
    )

    # 对训练集拟合预处理并转换
    X_train_processed = preprocessor.fit_transform(X_train)
    # 对测试集仅转换（使用训练集拟合的编码/填充参数）
    X_test_processed = preprocessor.transform(X_test)

    # 5. 使用SMOTE处理训练集中的类别不平衡
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    # 6. 随机森林模型与超参数网格
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)  # 不使用多进程

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }

    # 网格搜索（5折交叉验证，以ROC_AUC作为优化指标，不使用多进程）
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=1
    )

    # 训练模型（在过采样后的训练集上）
    grid_search.fit(X_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证ROC_AUC: {grid_search.best_score_:.4f}\n")

    # 7. 在测试集上评估
    y_pred = best_model.predict(X_test_processed)
    y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]  # 正类概率

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估指标
    print("========== 模型评估结果 ==========")
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score):  {f1:.4f}")
    print(f"AUC (ROC曲线下面积): {roc_auc:.4f}")
    print("==================================")


if __name__ == "__main__":
    main()