import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 忽略不必要的警告信息，保持控制台整洁
warnings.filterwarnings('ignore')


def parse_trust(val):
    """
    检验科专属数据清洗函数：
    处理LIS系统导出的TRUST滴度数据，兼容纯数字(如16)、比例字符串(如1:16)以及阴性结果。
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    # 处理阴性结果
    if val_str in ['阴性', 'Negative', 'neg', '-', '0']:
        return 0.0
    # 处理 "1:16" 或 "16" 格式
    parts = val_str.split(':')
    try:
        return float(parts[-1])
    except ValueError:
        return np.nan


def main():
    # ==========================================
    # 1. 数据加载与目标变量重构
    # ==========================================
    print("正在加载数据...")
    df = pd.read_csv('train_data.csv', encoding='utf-8')

    # 解析TRUST滴度并构建二分类目标变量 (>=16 为 1，否则为 0)
    df['TRUST_val'] = df['TRUST'].apply(parse_trust)
    df['TRUST_bin'] = (df['TRUST_val'] >= 16).astype(int)

    # 丢弃原始TRUST列和解析中间列，保留特征和二分类目标
    X = df.drop(columns=['TRUST', 'TRUST_val', 'TRUST_bin'])
    y = df['TRUST_bin']

    # ==========================================
    # 2. 划分训练集与测试集
    # ==========================================
    # 注意：必须在SMOTE之前划分，防止数据泄露（Data Leakage）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"数据集划分完成。训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    print(f"训练集正例(>=1:16)比例: {y_train.mean():.2%}")

    # ==========================================
    # 3. 构建数据预处理管道 (ColumnTransformer)
    # ==========================================
    # 定义特征列表
    num_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']
    cat_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    cat_ord_cols = ['TPPA']

    # 连续变量：中位数填充
    num_transformer = SimpleImputer(strategy='median')

    # 分类变量(独热)：处理未知类别，输出密集矩阵以兼容后续步骤
    cat_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 分类变量(序数)：TPPA编码，处理测试集中可能出现的未知类别
    cat_ord_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat_onehot', cat_onehot_transformer, cat_onehot_cols),
            ('cat_ord', cat_ord_transformer, cat_ord_cols)
        ]
    )

    # ==========================================
    # 4. 构建包含SMOTE和模型的完整Pipeline
    # ==========================================
    # 使用 imblearn 的 Pipeline，确保 SMOTE 仅在训练集的交叉验证折内应用
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # ==========================================
    # 5. 超参数调优 (GridSearchCV)
    # ==========================================
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10],
        'clf__min_samples_split': [2],
        'clf__min_samples_leaf': [1],
        'clf__class_weight': ['balanced']
    }

    print("开始进行超参数网格搜索（不使用多进程）...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',  # 临床二分类任务通常以AUC为主要优化指标
        cv=5,
        n_jobs=1,  # 严格遵循要求：不使用多进程
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    print(f"网格搜索完成！最佳参数组合: {grid_search.best_params_}")

    # ==========================================
    # 6. 模型评估
    # ==========================================
    best_model = grid_search.best_estimator_

    # 预测类别与预测概率
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 打印评估报告
    print("\n" + "=" * 40)
    print("随机森林模型测试集评估报告 (TRUST ≥ 1:16)")
    print("=" * 40)
    print(f"准确率 (Accuracy)  : {acc:.4f}")
    print(f"召回率 (Recall)    : {rec:.4f}  <-- 临床重点关注：避免漏诊高滴度患者")
    print(f"精确率 (Precision) : {prec:.4f}")
    print(f"F1分数 (F1-score)  : {f1:.4f}")
    print(f"AUC曲线下面积      : {auc:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
