"""
Train model từ file leaf_features.csv có sẵn
- Đọc CSV features
- Train SVM + RF với feature sets F1-F4
- Lưu kết quả và model
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# -----------------------------
# Chọn feature set
# -----------------------------
def prepare_Xy(df, feature_set):
    feature_cols = []
    if feature_set=='f1':
        feature_cols = [c for c in df.columns if c.startswith('f1_')]
    elif feature_set=='f2':
        feature_cols = [c for c in df.columns if c.startswith('f2_')]
    elif feature_set=='f3':
        feature_cols = [c for c in df.columns if c.startswith('f3_')]
    elif feature_set=='f4':
        feature_cols = [c for c in df.columns if c.startswith('f1_') or c.startswith('f2_') or c.startswith('f3_')]
    X = df[feature_cols].values
    y = df['label'].values
    return X, y

# -----------------------------
# Main
# -----------------------------
def main(features_file='features/leaf_features.csv'):
    os.makedirs('models', exist_ok=True)
    os.makedirs('features', exist_ok=True)
    
    # đọc CSV có sẵn
    df_features = pd.read_csv(features_file)
    
    
    feature_sets = ['f1','f2','f3','f4']
    model_types = ['svm','rf']
    results = []
    
    for feature_set in feature_sets:
        for model_type in model_types:
            df_train = df_features[df_features['split']=='train']
            df_val = df_features[df_features['split']=='val']
            df_test = df_features[df_features['split']=='test']

            print("Train:", len(df_train))
            print("Val:", len(df_val))
            print("Test:", len(df_test))
            
            X_train, y_train = prepare_Xy(df_train, feature_set)
            X_val, y_val = prepare_Xy(df_val, feature_set)
            X_test, y_test = prepare_Xy(df_test, feature_set)
            
            # Train model
            if model_type=='svm':
                model = SVC(kernel='rbf', probability=True, random_state=42)
            elif model_type=='rf':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            result = {
                'feature_set': feature_set,
                'model_type': model_type,
                'n_features': X_train.shape[1],
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'model_path': f'models/{model_type}_{feature_set}.pkl'
            }
            results.append(result)
            joblib.dump(model, result['model_path'])
    
    # Lưu kết quả
    df_results = pd.DataFrame(results)
    df_results.to_csv('features/results_summary.csv', index=False)
    print("Hoàn tất! Kết quả lưu ở:")
    print(" - features/results_summary.csv")
    print(" - models/*.pkl")

if __name__=='__main__':
    main('features/leaf_features.csv')