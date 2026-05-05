"""
Script 04: Train baseline models (Model 1A/1B)
Model 1A: Severity Classification (4 classes: 0,1,2,3)
Model 1B: Severity Regression (continuous severity_ratio)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
from tqdm import tqdm


def get_feature_columns(df, feature_set='f4'):
    """Lấy tên các cột feature tương ứng với feature_set."""
    if feature_set == 'f1':
        return [col for col in df.columns if col.startswith('f1_')]
    elif feature_set == 'f2':
        return [col for col in df.columns if col.startswith('f1_') or col.startswith('f2_')]
    elif feature_set == 'f3':
        return [col for col in df.columns if col.startswith('f1_') or col.startswith('f3_')]
    elif feature_set == 'f4':
        return [col for col in df.columns if col.startswith('f1_') or 
                col.startswith('f2_') or col.startswith('f3_')]
    else:
        raise ValueError(f"Feature set khong hop le: {feature_set}")


def prepare_data(df_features, feature_set='f4', split='train'):
    """Chuẩn bị dữ liệu X, y từ DataFrame features."""
    if split != 'all':
        # Nếu có cột 'split', filter theo split
        if 'split' in df_features.columns:
            df = df_features[df_features['split'] == split].copy()
        # Nếu có cột 'final_split', filter theo final_split
        elif 'final_split' in df_features.columns:
            df = df_features[df_features['final_split'] == split].copy()
        else:
            # Nếu không có cột split, dùng toàn bộ
            df = df_features.copy()
    else:
        df = df_features.copy()
    
    if len(df) == 0:
        return np.array([]), np.array([]), np.array([])
    
    feature_cols = get_feature_columns(df, feature_set)
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Thieu columns: {missing_cols[:5]}...")
    
    X = df[feature_cols].values
    y_severity = df['severity'].values
    y_ratio = df['severity_ratio'].values
    
    return X, y_severity, y_ratio


def train_classification_model(X_train, y_train, X_val, y_val, model_type='svm', 
                               feature_set_name='f4', use_grid_search=False):
    """
    Model 1A: Severity Classification
    """
    print(f"\nTraining {model_type.upper()} Classification voi {feature_set_name.upper()}")
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if model_type == 'svm':
        if use_grid_search:
            param_grid = {
                'C': [1, 10, 100],
                'gamma': ['scale', 0.001, 0.01]
            }
            model = GridSearchCV(
                SVC(kernel='rbf', random_state=42, probability=True),
                param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1
            )
            model.fit(X_train_scaled, y_train)
            model = model.best_estimator_
            print(f"Best params: {model.get_params()}")
        else:
            model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
            model.fit(X_train_scaled, y_train)
    
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)  # RF không cần scale
    
    elif model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            max_iter=200,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
    
    else:
        raise ValueError(f"Model type khong hop le: {model_type}")
    
    # Evaluate
    if model_type == 'rf':
        y_pred = model.predict(X_val)
    else:
        y_pred = model.predict(X_val_scaled)
    
    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    print(f"Val Accuracy: {accuracy:.4f}, F1-macro: {f1_macro:.4f}, F1-weighted: {f1_weighted:.4f}")
    
    return model, scaler if model_type != 'rf' else None


def train_regression_model(X_train, y_train, X_val, y_val, model_type='ridge',
                          feature_set_name='f4'):
    """
    Model 1B: Severity Regression
    """
    print(f"\nTraining {model_type.upper()} Regression voi {feature_set_name.upper()}")
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)
        X_val_use = X_val_scaled
    
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        X_val_use = X_val
    
    elif model_type == 'mlp':
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            max_iter=200,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        X_val_use = X_val_scaled
    
    else:
        raise ValueError(f"Model type khong hop le: {model_type}")
    
    # Evaluate
    y_pred = model.predict(X_val_use)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"Val MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return model, scaler if model_type != 'rf' else None


def train_all_baseline_models(features_file='data/leaf_features_with_severity.csv',
                             feature_sets=['f1', 'f2', 'f3', 'f4'],
                             model_types=['svm', 'rf', 'mlp'],
                             use_grid_search=False):
    """
    Train tất cả baseline models (1A và 1B).
    """
    # Đọc features
    df_features = pd.read_csv(features_file)
    
    # Chia train/val/test
    # Dùng split gốc từ dataset
    df_train = df_features[df_features['split'] == 'train'].copy()
    df_val = df_features[df_features['split'] == 'val'].copy()
    
    # Tách 20% từ val làm test
    # Tạo stratify column
    stratify_col = df_val[['severity', 'crop', 'disease']].apply(
        lambda x: f"{x['severity']}_{x['crop']}_{x['disease']}", axis=1
    )
    
    # Kiểm tra xem có đủ mẫu để chia test không
    if len(df_val) < 10:
        print(f"Warning: Val set qua nho ({len(df_val)}), dung toan bo val lam test")
        df_test = df_val.copy()
        df_val = pd.DataFrame()  # Empty val set
    else:
        min_samples = stratify_col.value_counts().min()
        if min_samples < 2:
            print(f"Warning: Khong du mau de stratify (min={min_samples}), chia khong stratify")
            df_val, df_test = train_test_split(
                df_val,
                test_size=0.2,
                random_state=42
            )
        else:
            df_val, df_test = train_test_split(
                df_val,
                test_size=0.2,
                random_state=42,
                stratify=stratify_col
            )
    
    # Gán final_split
    df_train['final_split'] = 'train'
    if len(df_val) > 0:
        df_val['final_split'] = 'val'
    df_test['final_split'] = 'test'
    
    dfs_to_concat = [df_train, df_test]
    if len(df_val) > 0:
        dfs_to_concat.insert(1, df_val)
    df_all = pd.concat(dfs_to_concat, ignore_index=True)
    
    print(f"\n{'='*70}")
    print("TRAIN BASELINE MODELS")
    print(f"{'='*70}")
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    results = []
    
    # Model 1A: Classification
    for feature_set in feature_sets:
        for model_type in model_types:
            try:
                print(f"\n{'#'*70}")
                print(f"Model 1A - Classification: {model_type.upper()} + {feature_set.upper()}")
                print(f"{'#'*70}")
                
                X_train, y_train_sev, _ = prepare_data(df_train, feature_set, 'all')
                
                # Sử dụng val nếu có, nếu không dùng test làm val
                if len(df_val) > 0:
                    X_val, y_val_sev, _ = prepare_data(df_val, feature_set, 'all')
                else:
                    # Dùng test làm val nếu không có val
                    X_test_temp, y_test_sev_temp, _ = prepare_data(df_test, feature_set, 'all')
                    X_val, y_val_sev = X_test_temp, y_test_sev_temp
                
                X_test, y_test_sev, _ = prepare_data(df_test, feature_set, 'all')
                
                # Train
                model, scaler = train_classification_model(
                    X_train, y_train_sev, X_val, y_val_sev,
                    model_type, feature_set, use_grid_search
                )
                
                # Evaluate on test
                if model_type == 'rf':
                    y_test_pred = model.predict(X_test)
                else:
                    X_test_scaled = scaler.transform(X_test)
                    y_test_pred = model.predict(X_test_scaled)
                
                test_acc = accuracy_score(y_test_sev, y_test_pred)
                test_f1_macro = f1_score(y_test_sev, y_test_pred, average='macro', zero_division=0)
                test_f1_weighted = f1_score(y_test_sev, y_test_pred, average='weighted', zero_division=0)
                
                # Lưu model
                model_path = f"models/{model_type}_f{feature_set[-1]}_classification.pkl"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(model, model_path)
                if scaler:
                    scaler_path = f"models/{model_type}_f{feature_set[-1]}_scaler.pkl"
                    joblib.dump(scaler, scaler_path)
                
                results.append({
                    'model_type': 'classification',
                    'algorithm': model_type,
                    'feature_set': feature_set,
                    'n_features': X_train.shape[1],
                    'test_accuracy': test_acc,
                    'test_f1_macro': test_f1_macro,
                    'test_f1_weighted': test_f1_weighted,
                    'model_path': model_path
                })
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Model 1B: Regression
    for feature_set in feature_sets:
        for model_type in ['ridge', 'rf', 'mlp']:
            try:
                print(f"\n{'#'*70}")
                print(f"Model 1B - Regression: {model_type.upper()} + {feature_set.upper()}")
                print(f"{'#'*70}")
                
                X_train, _, y_train_ratio = prepare_data(df_train, feature_set, 'all')
                
                # Sử dụng val nếu có, nếu không dùng test làm val
                if len(df_val) > 0:
                    X_val, _, y_val_ratio = prepare_data(df_val, feature_set, 'all')
                else:
                    # Dùng test làm val nếu không có val
                    X_test_temp, _, y_test_ratio_temp = prepare_data(df_test, feature_set, 'all')
                    X_val, y_val_ratio = X_test_temp, y_test_ratio_temp
                
                X_test, _, y_test_ratio = prepare_data(df_test, feature_set, 'all')
                
                # Train
                model, scaler = train_regression_model(
                    X_train, y_train_ratio, X_val, y_val_ratio,
                    model_type, feature_set
                )
                
                # Evaluate on test (chỉ nếu có test samples)
                if len(X_test) == 0:
                    print("Warning: Test set rong, bo qua evaluation")
                    continue
                
                if model_type == 'rf':
                    y_test_pred = model.predict(X_test)
                else:
                    X_test_scaled = scaler.transform(X_test)
                    y_test_pred = model.predict(X_test_scaled)
                
                test_mae = mean_absolute_error(y_test_ratio, y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test_ratio, y_test_pred))
                test_r2 = r2_score(y_test_ratio, y_test_pred)
                
                # Lưu model
                model_path = f"models/{model_type}_f{feature_set[-1]}_regression.pkl"
                joblib.dump(model, model_path)
                if scaler:
                    scaler_path = f"models/{model_type}_f{feature_set[-1]}_regression_scaler.pkl"
                    joblib.dump(scaler, scaler_path)
                
                results.append({
                    'model_type': 'regression',
                    'algorithm': model_type,
                    'feature_set': feature_set,
                    'n_features': X_train.shape[1],
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2,
                    'model_path': model_path
                })
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Lưu kết quả
    df_results = pd.DataFrame(results)
    results_file = 'results/baseline_results.csv'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    df_results.to_csv(results_file, index=False)
    
    print(f"\n{'='*70}")
    print("TONG KET KET QUA")
    print(f"{'='*70}")
    print(df_results.to_string(index=False))
    print(f"\nDa luu ket qua vao: {results_file}")
    
    return df_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--features-file', type=str, 
                       default='data/leaf_features_with_severity.csv',
                       help='File CSV chua features')
    parser.add_argument('--feature-sets', nargs='+',
                       choices=['f1', 'f2', 'f3', 'f4'],
                       default=['f1', 'f2', 'f3', 'f4'],
                       help='Cac feature set de train')
    parser.add_argument('--models', nargs='+',
                       choices=['svm', 'rf', 'mlp'],
                       default=['svm', 'rf', 'mlp'],
                       help='Cac model de train (classification)')
    parser.add_argument('--grid-search', action='store_true',
                       help='Dung GridSearchCV cho SVM')
    
    args = parser.parse_args()
    
    df_results = train_all_baseline_models(
        args.features_file,
        args.feature_sets,
        args.models,
        use_grid_search=args.grid_search
    )
