import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def main():
    # ==========================================
    # 1. 加载数据
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # ==========================================
    # 2. 特征与目标变量分离 & 目标二值化
    # ==========================================
    X = df.drop(columns=['TRUST'])
    # 目标列 TRUST 二值化：>=16 标记为 1 (正类)，<16 标记为 0 (负类)
    y = (df['TRUST'] >= 16).astype(int)

    # ==========================================
    # 3. 划分训练集和测试集
    # ==========================================
    # 使用 stratify=y 确保训练集和测试集中的正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 4. 定义特征列分组
    # ==========================================
    # 注：将您描述中的拼写错误 'DIAGONSIS' 修正为了正确的 'DIAGNOSIS'
    cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ordinal_cols = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # ==========================================
    # 5. 构建数据预处理器 (ColumnTransformer)
    # ==========================================
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量 (One-Hot)：独热编码，handle_unknown='ignore' 防止测试集出现未知类别报错
    cat_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量 (Ordinal)：序数编码，未知类别映射为 -1
    cat_ordinal_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat_onehot', cat_onehot_transformer, cat_onehot_cols),
            ('cat_ordinal', cat_ordinal_transformer, cat_ordinal_cols)
        ]
    )

    # ==========================================
    # 6. 构建完整的 Pipeline (包含 SMOTE 和 模型)
    # ==========================================
    # 使用 imblearn 的 Pipeline，确保 SMOTE 仅在训练集的 CV 折叠中应用，防止数据泄露
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 7. 超参数调优 (GridSearchCV)
    # ==========================================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("正在进行超参数网格搜索（不使用多进程）...")
    # n_jobs=1 确保不使用多进程；scoring='f1' 适合不平衡分类任务
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,
        verbose=1
    )

    # ==========================================
    # 8. 模型训练
    # ==========================================
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # ==========================================
    # 9. 模型评估
    # ==========================================
    print("正在测试集上评估模型...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 获取正类的概率用于计算 AUC

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # ==========================================
    # 10. 打印结果
    # ==========================================
    print("\n" + "=" * 40)
    print("模型评估结果")
    print("=" * 40)
    print(f"最佳超参数: {grid_search.best_params_}")
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"F1分数 (F1-score) : {f1:.4f}")
    print(f"AUC曲线下面积     : {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
