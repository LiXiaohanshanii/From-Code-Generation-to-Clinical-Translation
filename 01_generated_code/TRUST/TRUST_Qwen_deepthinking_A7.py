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

    # 确保目标列TRUST为数值类型，并构建二分类标签 (>=16 为 1，否则为 0)
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')
    y = (df['TRUST'] >= 16).astype(int)

    # 分离特征和目标
    X = df.drop(columns=['TRUST'])

    # ==========================================
    # 2. 定义特征列
    # ==========================================
    numeric_features = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    categorical_features_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_features_ord = ['TPPA']

    # ==========================================
    # 3. 构建数据预处理管道
    # ==========================================
    # 连续变量：中位数填充
    numeric_transformer = SimpleImputer(strategy='median')

    # 分类变量 (SEX, DEPT, DIAGNOSIS)：独热编码
    # sparse_output=False 确保输出为密集数组，兼容后续SMOTE
    categorical_transformer_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量 (TPPA)：序数编码
    # handle_unknown='use_encoded_value' 防止测试集中出现未见过的类别导致报错
    categorical_transformer_ord = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat_ohe', categorical_transformer_ohe, categorical_features_ohe),
            ('cat_ord', categorical_transformer_ord, categorical_features_ord)
        ]
    )

    # ==========================================
    # 4. 构建包含SMOTE和模型的完整管道
    # ==========================================
    # 使用 imblearn.pipeline.Pipeline 确保 SMOTE 仅在训练集的交叉验证折叠中应用，防止数据泄露
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 划分训练集与测试集
    # ==========================================
    print("正在划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 6. 超参数调优 (GridSearchCV)
    # ==========================================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("开始进行超参数调优（不使用多进程），这可能需要一些时间...")
    # n_jobs=1 确保不使用多进程
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # 针对不平衡数据，使用F1-score作为调优的主要评估指标
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"最佳超参数组合: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # ==========================================
    # 7. 模型评估
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
    print("       模型评估结果       ")
    print("=" * 30)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}")
    print(f"精确率 (Precision) : {prec:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积 (AUC): {auc:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
