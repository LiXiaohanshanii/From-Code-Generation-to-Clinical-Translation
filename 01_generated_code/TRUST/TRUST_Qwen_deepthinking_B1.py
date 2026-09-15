import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 导入 imbalanced-learn 用于处理分类不平衡
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # ================= 1. 数据加载与目标变量处理 =================
    print("正在加载数据...")
    # 读取数据，指定 utf-8 编码
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 目标列 TRUST 处理：兼容可能存在的 "1:16" 字符串格式，提取实际滴度数值
    if df['TRUST'].dtype == object:
        df['TRUST'] = df['TRUST'].astype(str).str.split(':').str[-1]
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')

    # 构建二分类目标：>=16 为 1（高滴度/阳性），<16 为 0
    y = (df['TRUST'] >= 16).astype(int)

    # ================= 2. 特征工程定义 =================
    # 修正提示中的拼写错误 (DIAGONSIS -> DIAGNOSIS)
    categorical_cols_onehot = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_cols_ordinal = ['TPPA']
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 提取特征矩阵 X
    features = categorical_cols_onehot + categorical_cols_ordinal + continuous_cols
    X = df[features]

    # 划分训练集和测试集 (分层抽样保证正负样本比例一致)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ================= 3. 构建数据预处理管道 =================
    # 连续变量：中位数填充缺失值
    continuous_transformer = SimpleImputer(strategy='median')

    # 分类变量 (独热编码)：处理未知类别以防测试集报错
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量 (序数编码)：TPPA 具有滴度顺序，处理未知类别
    ordinal_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', continuous_transformer, continuous_cols),
            ('cat_onehot', onehot_transformer, categorical_cols_onehot),
            ('cat_ordinal', ordinal_transformer, categorical_cols_ordinal)
        ]
    )

    # ================= 4. 构建包含 SMOTE 和分类器的完整管道 =================
    # 使用 imblearn 的 Pipeline，确保 SMOTE 只在 CV 的训练折中应用，防止数据泄露
    smote = SMOTE(random_state=42)
    rf_classifier = RandomForestClassifier(random_state=42)

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', rf_classifier)
    ])

    # ================= 5. 超参数调优 (GridSearchCV) =================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("开始进行超参数网格搜索（不使用多进程）...")
    # 对于不平衡数据，推荐使用 roc_auc 作为调优评估指标
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=1,  # 严格限制不使用多进程
        verbose=1
    )

    # 训练模型
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print(f"\n最佳超参数组合: {grid_search.best_params_}")
    print(f"交叉验证最佳 AUC 得分: {grid_search.best_score_:.4f}")

    # ================= 6. 模型评估 =================
    print("\n正在测试集上评估模型...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印结果
    print("-" * 30)
    print("模型评估指标 (测试集):")
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"召回率 (Recall)   : {rec:.4f}")
    print(f"F1分数 (F1-score) : {f1:.4f}")
    print(f"AUC 曲线下面积    : {auc:.4f}")
    print("-" * 30)


if __name__ == '__main__':
    main()
