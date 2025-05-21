import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 讀取資料
df = pd.read_csv('./3yscores.csv')

# 只保留常見學校、管道，其餘合併為「其他」
min_count = 20  # 門檻可調整

# 處理學校
school_counts = df['學校'].value_counts()
common_schools = school_counts[school_counts >= min_count].index
df['學校_clean'] = df['學校'].where(df['學校'].isin(common_schools), other='其他')

# 處理入學管道
method_counts = df['入學管道'].value_counts()
common_methods = method_counts[method_counts >= min_count].index
df['入學管道_clean'] = df['入學管道'].where(df['入學管道'].isin(common_methods), other='其他')

# 欄位名稱
school_col = '學校_clean'
method_col = '入學管道_clean'
feature_cols = [
    '10ASub1', '10ASub2', '10ASub3',  # 高一上學期
    '10BSub1', '10BSub2', '10BSub3',  # 高一下學期
    '11ASub1', '11ASub2', '11ASub3',  # 高二上學期
    '11BSub1', '11BSub2', '11BSub3'   # 高二下學期
]

# 去除缺漏值
df = df.dropna(subset=feature_cols + [school_col, method_col])

# Label encoding
school_encoder = LabelEncoder()
method_encoder = LabelEncoder()
df['school_label'] = school_encoder.fit_transform(df[school_col])
df['method_label'] = method_encoder.fit_transform(df[method_col])

# 分割資料
X = df[feature_cols]
y_school = df['school_label']
y_method = df['method_label']

def train_and_evaluate_model(X, y, target_name, encoder):
    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定義XGBoost參數網格
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # 創建XGBoost分類器
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # 使用網格搜索找到最佳參數
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"\n訓練{target_name}預測模型...")
    grid_search.fit(X_train, y_train)
    
    # 使用最佳參數的模型
    best_model = grid_search.best_estimator_
    
    # 交叉驗證
    cv_scores = cross_val_score(best_model, X, y, cv=5)
    print(f"\n{target_name}模型交叉驗證分數：")
    print(f"平均準確率: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # 在測試集上評估
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    print(f"\n{target_name}預測結果：")
    print(classification_report(
        y_test,
        y_pred,
        labels=best_model.classes_,
        target_names=encoder.inverse_transform(best_model.classes_),
        zero_division=0
    ))
    
    # 特徵重要性分析
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'{target_name}預測模型 - 特徵重要性')
    plt.tight_layout()
    plt.savefig(f'{target_name}_feature_importance.png')
    plt.close()
    
    return best_model, feature_importance

# 訓練和評估學校預測模型
school_model, school_importance = train_and_evaluate_model(X, y_school, "學校", school_encoder)

# 訓練和評估入學管道預測模型
method_model, method_importance = train_and_evaluate_model(X, y_method, "入學管道", method_encoder)

# 輸出最佳模型參數
print("\n學校預測模型最佳參數：")
print(school_model.get_params())
print("\n入學管道預測模型最佳參數：")
print(method_model.get_params())

# 保存模型
joblib.dump(school_model, 'models/school_model.joblib')
joblib.dump(method_model, 'models/method_model.joblib')
joblib.dump(school_encoder, 'models/school_encoder.joblib')
joblib.dump(method_encoder, 'models/method_encoder.joblib')
