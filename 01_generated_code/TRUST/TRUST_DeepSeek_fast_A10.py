import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')


# ==================== 数据加载 ====================
def load_data(file_path='train_data.csv'):
    """加载训练数据"""
    df = pd.read_csv(file_path, encoding='utf-8')
    return df


# ==================== 数据预处理 ====================
def create_preprocessor():
    """
    创建数据预处理转换器
    分类变量：
        - SEX, DEPT, DIAGNOSIS: 独热编码 (OneHotEncoder)
        - TPPA: 序数编码 (OrdinalEncoder)
    连续变量：中位数填充 (SimpleImputer)
    """
    # 定义特征列
    categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
    categorical_ordinal_cols = ['TPPA']
    numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

    # 创建预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_onehot_cols),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan),
             categorical_ordinal_cols),
            ('num', SimpleImputer(strategy='median'), numerical_cols)
        ],
        remainder='drop'  # 丢弃未指定的列（如目标列）
    )

    return preprocessor


# ==================== 模型构建与评估 ====================
def train_and_evaluate_model(df, target_col='TRUST'):
    """
    使用随机森林分类器训练并评估模型
    """
    # 分离特征和目标变量
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 二分类：将目标值转换为 0 (True < 16) 和 1 (True >= 16)
    y_binary = (y >= 16).astype(int)

    # 划分训练集和测试集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    print(f"训练集类别分布:\n{y_train.value_counts()}")
    print(f"测试集类别分布:\n{y_test.value_counts()}")

    # 创建预处理转换器
    preprocessor = create_preprocessor()

    # 定义随机森林分类器
    rf_classifier = RandomForestClassifier(
        random_state=42,
        n_jobs=1  # 不使用多进程
    )

    # 创建包含SMOTE的完整流水线
    # 注意：SMOTE应在预处理之后、模型训练之前应用
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', rf_classifier)
    ])

    # 定义超参数搜索空间
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [1],
        'classifier__class_weight': ['balanced']
    }

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # 5折交叉验证
        scoring='f1',  # 使用F1分数作为优化指标
        n_jobs=1,  # 不使用多进程
        verbose=1
    )

    print("\n开始超参数调优...")
    grid_search.fit(X_train, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

    # 使用最佳模型进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 打印评估结果
    print("\n" + "=" * 50)
    print("模型评估结果 (测试集)")
    print("=" * 50)
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"F1分数 (F1-score):  {f1:.4f}")
    print(f"AUC:               {auc:.4f}")
    print("=" * 50)

    # 返回模型和评估指标
    results = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'auc': auc,
        'best_params': grid_search.best_params_,
        'model': best_model
    }

    return results


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 加载数据
    print("加载数据...")
    df = load_data('train_data.csv')
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"目标列唯一值: {df['TRUST'].unique()}")

    # 运行模型训练和评估
    results = train_and_evaluate_model(df)

    # 可选：输出特征重要性（如果需要在最佳模型上查看）
    # 注意：由于流水线中包含预处理，获取特征重要性需要从流水线中提取分类器
    best_model = results['model']
    # 获取预处理后的特征名称（仅用于展示）
    try:
        # 获取独热编码后的特征名称
        preprocessor = best_model.named_steps['preprocessor']
        onehot_encoder = preprocessor.named_transformers_['onehot']
        ordinal_encoder = preprocessor.named_transformers_['ordinal']

        # 获取原始特征
        categorical_onehot_cols = ['SEX', 'DEPT', 'DIAGNOSIS']
        categorical_ordinal_cols = ['TPPA']
        numerical_cols = ['AGE', 'TP', 'HIV', 'WBC', 'RBC', 'PLT', 'NC', 'LY', 'NLR']

        # 获取独热编码后的特征名称
        onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_onehot_cols)
        # 序数编码特征和数值特征名称保持不变
        ordinal_feature_names = categorical_ordinal_cols
        numerical_feature_names = numerical_cols

        all_feature_names = list(onehot_feature_names) + ordinal_feature_names + numerical_feature_names

        # 获取随机森林分类器
        rf_clf = best_model.named_steps['classifier']

        # 打印前10个最重要的特征
        importances = rf_clf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n" + "=" * 50)
        print("特征重要性 (Top 10)")
        print("=" * 50)
        for i in range(min(10, len(all_feature_names))):
            idx = indices[i]
            print(f"{i + 1}. {all_feature_names[idx]}: {importances[idx]:.4f}")
    except Exception as e:
        print(f"\n无法提取特征重要性: {e}")

    print("\n程序执行完成！")