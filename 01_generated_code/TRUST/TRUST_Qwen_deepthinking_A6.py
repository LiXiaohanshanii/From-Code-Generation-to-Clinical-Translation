import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def main():
    # ==========================================
    # 1. 加载数据与目标变量处理
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 目标列是最后一列 (TRUST)
    target_col = df.columns[-1]

    # 确保目标列为数值类型，并进行二分类转换 (>=16 为 1，否则为 0)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])  # 丢弃目标列无法转换为数值的异常行
    df['target_binary'] = (df[target_col] >= 16).astype(int)

    # 分离特征和目标
    X = df.drop(columns=[target_col, 'target_binary'])
    y = df['target_binary']

    # ==========================================
    # 2. 划分训练集和测试集
    # ==========================================
    print("正在划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 定义数据预处理步骤
    # ==========================================
    # 定义各类特征列
    cat_ohe_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ord_cols = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 数值变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量 (独热编码)：处理未知类别
    cat_ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量 (序数编码)：处理未知类别
    cat_ord_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat_ohe', cat_ohe_transformer, cat_ohe_cols),
            ('cat_ord', cat_ord_transformer, cat_ord_cols)
        ]
    )

    # ==========================================
    # 4. 构建 Pipeline (包含预处理、SMOTE和模型)
    # ==========================================
    # 使用 imblearn 的 Pipeline 确保 SMOTE 仅在训练集的每个 CV fold 中应用，防止数据泄露
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    print("正在进行超参数调优（不使用多进程），请耐心等待...")
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # n_jobs=1 确保不使用多进程
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n最佳超参数组合: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # ==========================================
    # 6. 模型评估
    # ==========================================
    print("正在评估模型...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "=" * 30)
    print("       模型评估结果")
    print("=" * 30)
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"F1分数 (F1-score) : {f1:.4f}")
    print(f"AUC曲线下面积     : {auc:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
