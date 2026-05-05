"""
Module training và evaluation cho các classifier.
Hỗ trợ SVM, Random Forest và các bộ feature F1-F4.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os
from pathlib import Path


def get_feature_columns(df, feature_set='f4'):
    """
    Lấy tên các cột feature tương ứng với feature_set.
    
    Args:
        df: DataFrame chứa features
        feature_set: 'f1', 'f2', 'f3', hoặc 'f4'
    
    Returns:
        List tên cột feature
    """
    if feature_set == 'f1':
        # F1: chỉ color features (12 features)
        return [col for col in df.columns if col.startswith('f1_')]
    elif feature_set == 'f2':
        # F2: F1 + vegetation (12 + 7 = 19 features)
        return [col for col in df.columns if col.startswith('f1_') or col.startswith('f2_')]
    elif feature_set == 'f3':
        # F3: F1 + texture (12 + 63 = 75 features)
        return [col for col in df.columns if col.startswith('f1_') or col.startswith('f3_')]
    elif feature_set == 'f4':
        # F4: F1 + F2 + F3 (12 + 7 + 63 = 82 features)
        return [col for col in df.columns if col.startswith('f1_') or 
                col.startswith('f2_') or col.startswith('f3_')]
    else:
        raise ValueError(f"Feature set không hợp lệ: {feature_set}")


def prepare_data(df_features, feature_set='f4', split='train'):
    """
    Chuẩn bị dữ liệu X, y từ DataFrame features.
    
    Args:
        df_features: DataFrame chứa features và split
        feature_set: 'f1', 'f2', 'f3', hoặc 'f4'
        split: 'train', 'val', 'test', hoặc 'all'
    
    Returns:
        X, y (numpy arrays)
    """
    # Lọc theo split
    if split != 'all':
        df = df_features[df_features['split'] == split].copy()
    else:
        df = df_features.copy()
    
    # Lấy feature columns
    feature_cols = get_feature_columns(df, feature_set)
    
    # Kiểm tra columns có tồn tại
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu columns: {missing_cols[:5]}...")
    
    X = df[feature_cols].values
    y = df['label'].values
    
    return X, y


def create_model(model_type='svm', use_scaler=True):
    """
    Tạo model pipeline với scaler (nếu cần).
    
    Args:
        model_type: 'svm', 'rf', hoặc 'mlp'
        use_scaler: Có dùng StandardScaler không
    
    Returns:
        sklearn Pipeline hoặc model
    """
    if model_type == 'svm':
        model = SVC(kernel='rbf', probability=True, random_state=42)
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_type == 'mlp':
        model = MLPClassifier(random_state=42, max_iter=500)
    else:
        raise ValueError(f"Model type không hợp lệ: {model_type}")
    
    if use_scaler and model_type != 'rf':  # RF không cần scale
        return Pipeline([('scaler', StandardScaler()), ('clf', model)])
    else:
        return model


def get_param_grid(model_type='svm'):
    """
    Trả về parameter grid cho GridSearchCV.
    
    Args:
        model_type: 'svm', 'rf', hoặc 'mlp'
    
    Returns:
        dict parameter grid
    """
    if model_type == 'svm':
        return {
            'clf__C': [1, 10, 100],
            'clf__gamma': ['scale', 'auto', 0.001, 0.01]
        }
    elif model_type == 'rf':
        return {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'mlp':
        return {
            'clf__hidden_layer_sizes': [(64,), (128,), (128, 64)],
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__learning_rate': ['constant', 'adaptive']
        }
    else:
        raise ValueError(f"Model type không hợp lệ: {model_type}")


def train_model(X_train, y_train, X_val, y_val, model_type='svm', 
                feature_set_name='f4', use_grid_search=True, n_jobs=-1):
    """
    Train model với hyperparameter tuning.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (dùng để chọn best model)
        model_type: 'svm', 'rf', hoặc 'mlp'
        feature_set_name: Tên feature set (để lưu model)
        use_grid_search: Dùng GridSearchCV hay train trực tiếp
        n_jobs: Số jobs cho GridSearchCV
    
    Returns:
        Trained model (best model)
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} với {feature_set_name.upper()}")
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    print(f"{'='*60}")
    
    # Tạo model
    use_scaler = (model_type != 'rf')
    model = create_model(model_type, use_scaler)
    
    if use_grid_search:
        # GridSearchCV
        param_grid = get_param_grid(model_type)
        
        # Điều chỉnh param_grid nếu không dùng scaler (model không phải Pipeline)
        if not isinstance(model, Pipeline):
            # Nếu model không phải Pipeline, bỏ prefix 'clf__'
            param_grid = {k.replace('clf__', ''): v for k, v in param_grid.items()}
        
        print(f"Đang tìm hyperparameters tốt nhất...")
        print(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,  # 3-fold CV trên train
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        
        # Train trên train set
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
    else:
        # Train trực tiếp với default params
        print("Training với default parameters...")
        best_model = model
        best_model.fit(X_train, y_train)
    
    # Đánh giá trên validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  F1-macro: {val_f1:.4f}")
    
    return best_model


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Đánh giá model trên test set.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        class_names: List tên lớp (để hiển thị)
    
    Returns:
        dict chứa các metrics
    """
    print(f"\n{'='*60}")
    print("Evaluation trên Test Set")
    print(f"{'='*60}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\nTest Results:")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  Precision (macro):   {precision_macro:.4f}")
    print(f"  Recall (macro):      {recall_macro:.4f}")
    print(f"  F1-score (macro):    {f1_macro:.4f}")
    print(f"  Precision (weighted): {precision_weighted:.4f}")
    print(f"  Recall (weighted):    {recall_weighted:.4f}")
    print(f"  F1-score (weighted):  {f1_weighted:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    if class_names:
        print(f"\nClassification Report:")
        # Chỉ lấy các lớp có trong dữ liệu
        unique_labels = sorted(np.unique(y_test))
        available_class_names = [class_names[i] for i in unique_labels if i < len(class_names)]
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=available_class_names, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def save_model(model, model_path, feature_set_name, model_type):
    """
    Lưu model vào file.
    
    Args:
        model: Trained model
        model_path: Đường dẫn file để lưu
        feature_set_name: Tên feature set
        model_type: Loại model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nĐã lưu model vào: {model_path}")


def load_model(model_path):
    """
    Load model từ file.
    
    Args:
        model_path: Đường dẫn file model
    
    Returns:
        Loaded model
    """
    return joblib.load(model_path)


if __name__ == '__main__':
    # Test script
    print("Module train_eval.py - Test")
    print("Để sử dụng, import và gọi các hàm từ module này.")
