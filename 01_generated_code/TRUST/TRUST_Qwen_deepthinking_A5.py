import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
# 注意：必须使用 imblearn 的 Pipeline，因为 sklearn 的原生 Pipeline 不支持中间步骤改变样本数量（如SMOTE）
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # ==========================================
    # 1. 加载数据与目标变量处理
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 提取特征 (假设除了TRUST外均为特征列，或显式丢弃TRUST)
    X = df.drop(columns=['TRUST'])

    # 目标列处理：预测 TRUST 是否 >= 16 (二分类：1表示>=16，0表示<16)
    y = (df['TRUST'] >= 16).astype(int)

    # ==========================================
    # 2. 划分训练集和测试集
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. 数据预处理配置
    # ==========================================
    # 定义各类特征列
    continuous_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    ordinal_cols = ['TPPA']

    # 连续变量：中位数填充
    continuous_transformer = SimpleImputer(strategy='median')

    # 分类变量 (独热编码)：处理未知类别以防测试集报错
    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量 (序数编码)：处理未知类别
    ordinal_transformer = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', continuous_transformer, continuous_cols),
            ('onehot', onehot_transformer, onehot_cols),
            ('ordinal', ordinal_transformer, ordinal_cols)
        ],
        remainder='drop'  # 丢弃未明确指定的其他列
    )

    # ==========================================
    # 4. 构建包含 SMOTE 和 模型的 Pipeline
    # ==========================================
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),  # 分类不平衡处理
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("开始进行超参数调优与模型训练 (单进程模式)...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # 针对不平衡数据，使用 F1-score 作为调优指标
        n_jobs=1,  # 明确指定不使用多进程
        verbose=1
    )

    # 训练模型
    grid_search.fit(X_train, y_train)

    # ==========================================
    # 6. 模型评估
    # ==========================================
    best_model = grid_search.best_estimator_

    # 预测
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取正类概率用于计算AUC

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印结果
    print("\n" + "=" * 40)
    print("模型训练与评估完成！")
    print("=" * 40)
    print(f"最佳超参数组合: {grid_search.best_params_}")
    print("-" * 40)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}")
    print(f"精确率 (Precision) : {prec:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积 (AUC): {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
