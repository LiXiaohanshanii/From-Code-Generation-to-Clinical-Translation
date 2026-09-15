import pandas as pd
import numpy as np

# 机器学习与数据预处理库
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

# 不平衡数据处理库 (需安装 imbalanced-learn: pip install imbalanced-learn)
from imblearn.over_sampling import SMOTE


def build_and_evaluate_rf():
    # ---------------------------------------------------------
    # 1. 数据读取与目标变量构建
    # ---------------------------------------------------------
    # 读取数据集，utf-8 编码
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 构建二分类目标：TRUST 滴度 >= 16 记为 1，否则记为 0
    # 注意：确保 TRUST 列已解析为数值型
    df['TARGET'] = (pd.to_numeric(df['TRUST'], errors='coerce') >= 16).astype(int)

    # 定义列类型
    cat_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']  # 独热编码变量
    cat_ord_cols = ['TPPA']  # 序数编码变量
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']  # 连续变量

    X = df[cat_ohe_cols + cat_ord_cols + num_cols]
    y = df['TARGET']

    # 划分训练集与测试集 (80% 训练, 20% 测试，分层抽样保持正负样本比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # 2. 数据预处理管道构建
    # ---------------------------------------------------------
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量：独热编码（适应不同版本的 sparse 参数设置）
    ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量：序数编码
    ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 使用 ColumnTransformer 组装预处理规则
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat_ohe', ohe_transformer, cat_ohe_cols),
            ('cat_ord', ord_transformer, cat_ord_cols)
        ]
    )

    # 在训练集上拟合预处理模型，并转换训练集与测试集
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # ---------------------------------------------------------
    # 3. 类别不平衡处理 (SMOTE)
    # ---------------------------------------------------------
    # 仅对训练集进行 SMOTE 重采样，避免测试集数据泄漏
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_prep, y_train)

    # ---------------------------------------------------------
    # 4. 随机森林模型构建与网格搜索调优
    # ---------------------------------------------------------
    # 参数网格（注：将参数中的中文 '平衡' 映射为 scikit-learn 标准参数 'balanced'）
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'class_weight': ['balanced']
    }

    rf_clf = RandomForestClassifier(random_state=42)

    # 采用 5 折交叉验证调优，设置 n_jobs=1（禁用多进程）
    grid_search = GridSearchCV(
        estimator=rf_clf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1
    )

    grid_search.fit(X_train_resampled, y_train_resampled)
    best_rf = grid_search.best_estimator_

    # ---------------------------------------------------------
    # 5. 模型评估
    # ---------------------------------------------------------
    # 在独立测试集上进行预测
    y_pred = best_rf.predict(X_test_prep)
    y_proba = best_rf.predict_proba(X_test_prep)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # ---------------------------------------------------------
    # 6. 输出结果
    # ---------------------------------------------------------
    print("=" * 40)
    print("      随机森林模型评估结果（检验科）    ")
    print("=" * 40)
    print(f"最佳超参数组合 : {grid_search.best_params_}")
    print("-" * 40)
    print(f"准确率 (Accuracy)   : {accuracy:.4f}")
    print(f"召回率 (Recall)     : {recall:.4f}")
    print(f"精确率 (Precision)  : {precision:.4f}")
    print(f"F1 分数 (F1-score)  : {f1:.4f}")
    print(f"ROC-AUC 面积 (AUC)  : {auc:.4f}")
    print("=" * 40)


if __name__ == '__main__':
    build_and_evaluate_rf()