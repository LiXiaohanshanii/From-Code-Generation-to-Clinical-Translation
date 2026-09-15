import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
# 注意：这里必须使用 imblearn 的 Pipeline，以支持 SMOTE 改变样本数量的操作
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # ==========================================
    # 1. 加载数据与目标变量转换
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 确保 TRUST 列为数值类型（防止数据中包含字符串格式的滴度如 "1:16"）
    # 如果是 "1:16" 格式，需先提取数字，这里假设已经是纯数字 1, 2, 4, 8, 16...
    df['TRUST'] = pd.to_numeric(df['TRUST'], errors='coerce')

    # 目标变量二分类转换：>=16 为 1 (阳性/高滴度)，<16 为 0
    df['TRUST_binary'] = (df['TRUST'] >= 16).astype(int)

    # 定义特征列
    # 注：提示词中出现了 DIAGNOSIS 和 DIAGONSIS，这里统一使用正确的拼写 DIAGNOSIS
    cat_cols_ohe = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_cols_ord = ['TPPA']
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 划分特征 (X) 和 目标变量 (y)
    X = df[cat_cols_ohe + cat_cols_ord + num_cols]
    y = df['TRUST_binary']

    # ==========================================
    # 2. 划分训练集和测试集
    # ==========================================
    # 分层抽样 (stratify=y) 确保训练集和测试集中正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"数据集划分完成 -> 训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

    # ==========================================
    # 3. 构建数据预处理管道 (ColumnTransformer)
    # ==========================================
    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量 (One-Hot)：独热编码，handle_unknown='ignore' 防止测试集出现未知类别报错
    cat_ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量 (Ordinal)：序数编码。
    # 注：TPPA如果是定性(阴/阳)或滴度，默认按字母/字典序编码。若需指定临床顺序，可传入 categories 参数。
    cat_ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat_ohe', cat_ohe_transformer, cat_cols_ohe),
            ('cat_ord', cat_ord_transformer, cat_cols_ord)
        ]
    )

    # ==========================================
    # 4. 构建包含 SMOTE 和 随机森林 的完整 Pipeline
    # ==========================================
    # 使用 imblearn.pipeline 确保 SMOTE 仅在训练集的 CV 折叠中应用，防止数据泄露
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    # 定义参数网格（注意需要加上 'classifier__' 前缀）
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    print("开始进行超参数网格搜索 (不使用多进程)...")
    # n_jobs=1 表示不使用多进程；scoring='f1' 因为是不平衡数据，F1比Accuracy更具参考价值
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n最佳超参数组合: {grid_search.best_params_}")

    # ==========================================
    # 6. 模型评估
    # ==========================================
    print("正在测试集上评估模型...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 获取预测为类别1的概率，用于计算AUC

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 40)
    print("🌟 随机森林模型测试集评估结果 🌟")
    print("=" * 40)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}")
    print(f"精确率 (Precision) : {prec:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积      : {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
