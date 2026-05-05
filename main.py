"""
Script chính để chạy toàn bộ pipeline Nhánh 2:
1. Index và chia dữ liệu
2. Trích xuất features (F1-F4)
3. Train và evaluate các model với từng bộ feature
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Fix encoding cho Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset_utils import build_image_index, split_dataset, CLASS_MAPPING, LABEL_TO_CLASS
from feature_extraction import extract_all_features
from train_eval import (
    prepare_data, train_model, evaluate_model, 
    save_model, get_feature_columns
)


def step1_index_data(data_dir='.', output_file='data/images_index.csv'):
    """
    Bước 1: Index dữ liệu và chia train/val/test.
    """
    print("\n" + "="*70)
    print("BUOC 1: INDEX VA CHIA DU LIEU")
    print("="*70)
    
    # Index dữ liệu
    df = build_image_index(data_dir)
    
    # Chia train/val/test
    df_split = split_dataset(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # Lưu
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_split.to_csv(output_file, index=False)
    print(f"\nDa luu index vao: {output_file}")
    
    return df_split


def step2_extract_features(index_file='image_index.csv', 
                          output_file='features/leaf_features.csv',
                          resize_size=None, sample_size=None):
    """
    Bước 2: Trích xuất features cho tất cả ảnh.
    
    Args:
        index_file: File CSV chứa index ảnh
        output_file: File CSV để lưu features
        resize_size: Tuple (H, W) để resize ảnh (None = giữ nguyên)
        sample_size: Số ảnh để test (None = tất cả)
    """
    print("\n" + "="*70)
    print("BUOC 2: TRICH XUAT FEATURES")
    print("="*70)
    
    # Đọc index
    df_index = pd.read_csv(index_file)
    
    # Sample nếu cần (để test nhanh) - stratified sample theo split để đảm bảo có đủ các lớp
    if sample_size and sample_size < len(df_index):
        print(f"Warning: Chi xu ly {sample_size} anh (stratified sample)")
        from sklearn.model_selection import train_test_split
        # Stratified sample theo cả label và split để giữ tỉ lệ
        # Lấy tỉ lệ từ mỗi split
        df_sampled = []
        for split_name in df_index['split'].unique():
            df_split = df_index[df_index['split'] == split_name].copy()
            split_ratio = len(df_split) / len(df_index)
            n_samples_split = max(1, int(sample_size * split_ratio))
            if n_samples_split < len(df_split):
                df_split_sampled, _ = train_test_split(
                    df_split,
                    train_size=n_samples_split,
                    stratify=df_split['label'],
                    random_state=42
                )
            else:
                df_split_sampled = df_split
            df_sampled.append(df_split_sampled)
        df_index = pd.concat(df_sampled, ignore_index=True)
    
    # Trích xuất features cho từng ảnh
    all_features = []
    failed_images = []
    
    print(f"\nĐang trích xuất features cho {len(df_index)} ảnh...")
    
    for idx, row in tqdm(df_index.iterrows(), total=len(df_index), desc="Extracting"):
        try:
            features_dict = extract_all_features(row['image_path'], resize_size=resize_size)
            
            # Tạo record
            record = {
                'image_path': row['image_path'],
                'class_name': row['class_name'],
                'label': row['label'],
                'split': row.get('split', 'unknown')
            }
            
            # Thêm F1 features
            for i, name in enumerate(features_dict['f1_names']):
                record[f'f1_{name}'] = features_dict['f1_features'][i]
            
            # Thêm F2 features (chỉ phần vegetation, F1 đã có)
            f2_start_idx = len(features_dict['f1_features'])
            for i, name in enumerate(features_dict['f2_names'][f2_start_idx:]):
                record[f'f2_{name}'] = features_dict['f2_features'][f2_start_idx + i]
            
            # Thêm F3 features (chỉ phần texture, F1 đã có)
            f3_start_idx = len(features_dict['f1_features'])
            for i, name in enumerate(features_dict['f3_names'][f3_start_idx:]):
                record[f'f3_{name}'] = features_dict['f3_features'][f3_start_idx + i]
            
            all_features.append(record)
            
        except Exception as e:
            print(f"\nWarning: Loi khi xu ly {row['image_path']}: {e}")
            failed_images.append(row['image_path'])
            continue
    
    if failed_images:
        print(f"\nWarning: Co {len(failed_images)} anh loi")
    
    # Tạo DataFrame
    df_features = pd.DataFrame(all_features)
    
    # Lưu
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_features.to_csv(output_file, index=False)
    
    print(f"\nDa luu features vao: {output_file}")
    print(f"  Số lượng ảnh: {len(df_features)}")
    print(f"  Số feature F1: {len([c for c in df_features.columns if c.startswith('f1_')])}")
    print(f"  Số feature F2: {len([c for c in df_features.columns if c.startswith('f2_')])}")
    print(f"  Số feature F3: {len([c for c in df_features.columns if c.startswith('f3_')])}")
    
    return df_features


def step3_train_evaluate(features_file='features/leaf_features.csv',
                         feature_sets=['f1', 'f2', 'f3', 'f4'],
                         model_types=['svm', 'rf'],
                         use_grid_search=True):
    """
    Bước 3: Train và evaluate các model với từng bộ feature.
    
    Args:
        features_file: File CSV chứa features
        feature_sets: List các feature set để train ['f1', 'f2', 'f3', 'f4']
        model_types: List các model type ['svm', 'rf', 'mlp']
        use_grid_search: Có dùng GridSearchCV không
    """
    print("\n" + "="*70)
    print("BUOC 3: TRAIN VA EVALUATE MODELS")
    print("="*70)
    
    # Đọc features
    df_features = pd.read_csv(features_file)
    
    # Class names
    class_names = [LABEL_TO_CLASS[i] for i in sorted(LABEL_TO_CLASS.keys())]
    
    # Kết quả
    results = []
    
    # Lặp qua từng combination
    for feature_set in feature_sets:
        for model_type in model_types:
            print(f"\n{'#'*70}")
            print(f"Feature Set: {feature_set.upper()} | Model: {model_type.upper()}")
            print(f"{'#'*70}")
            
            try:
                # Chuẩn bị dữ liệu
                X_train, y_train = prepare_data(df_features, feature_set, 'train')
                X_val, y_val = prepare_data(df_features, feature_set, 'val')
                X_test, y_test = prepare_data(df_features, feature_set, 'test')
                
                print(f"Feature shape: {X_train.shape[1]}")
                
                # Train
                model = train_model(
                    X_train, y_train, X_val, y_val,
                    model_type=model_type,
                    feature_set_name=feature_set,
                    use_grid_search=use_grid_search
                )
                
                # Evaluate trên test
                eval_results = evaluate_model(model, X_test, y_test, class_names)
                
                # Lưu model
                model_filename = f"{model_type}_{feature_set}.pkl"
                model_path = f"models/{model_filename}"
                save_model(model, model_path, feature_set, model_type)
                
                # Lưu kết quả
                result = {
                    'feature_set': feature_set,
                    'model_type': model_type,
                    'n_features': X_train.shape[1],
                    'accuracy': eval_results['accuracy'],
                    'precision_macro': eval_results['precision_macro'],
                    'recall_macro': eval_results['recall_macro'],
                    'f1_macro': eval_results['f1_macro'],
                    'precision_weighted': eval_results['precision_weighted'],
                    'recall_weighted': eval_results['recall_weighted'],
                    'f1_weighted': eval_results['f1_weighted'],
                    'model_path': model_path
                }
                results.append(result)
                
            except Exception as e:
                print(f"\nError: Loi khi train {model_type} voi {feature_set}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Lưu bảng kết quả
    df_results = pd.DataFrame(results)
    results_file = 'features/results_summary.csv'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    df_results.to_csv(results_file, index=False)
    
    print(f"\n{'='*70}")
    print("TỔNG KẾT KẾT QUẢ")
    print(f"{'='*70}")
    print(df_results.to_string(index=False))
    print(f"\nDa luu ket qua vao: {results_file}")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(description='Pipeline Nhánh 2 - Tea Leaf Disease Classification')
    parser.add_argument('--data-dir', type=str, default='.',
                       help='Thư mục chứa dữ liệu ảnh (mặc định: thư mục hiện tại)')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 0],
                       help='Bước để chạy: 1=index, 2=extract, 3=train, 0=all (mặc định: 0)')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('H', 'W'),
                       help='Resize ảnh về kích thước HxW (ví dụ: --resize 256 256)')
    parser.add_argument('--sample-size', type=int,
                       help='Số ảnh để test (chỉ dùng cho step 2)')
    parser.add_argument('--feature-sets', nargs='+', 
                       choices=['f1', 'f2', 'f3', 'f4'], 
                       default=['f1', 'f2', 'f3', 'f4'],
                       help='Các feature set để train (mặc định: tất cả)')
    parser.add_argument('--models', nargs='+',
                       choices=['svm', 'rf', 'mlp'],
                       default=['svm', 'rf'],
                       help='Các model để train (mặc định: svm, rf)')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='Không dùng GridSearchCV (train với default params)')
    
    args = parser.parse_args()
    
    # Resize size
    resize_size = tuple(args.resize) if args.resize else None
    
    # Step
    step = args.step if args.step is not None else 0
    
    print("\n" + "="*70)
    print("PIPELINE NHANH 2 - TEA LEAF DISEASE CLASSIFICATION")
    print("="*70)
    
    # Chạy các bước
    if step == 0 or step == 1:
        df_index = step1_index_data(args.data_dir)
    
    if step == 0 or step == 2:
        df_features = step2_extract_features(
            resize_size=resize_size,
            sample_size=args.sample_size
        )
    
    if step == 0 or step == 3:
        df_results = step3_train_evaluate(
            feature_sets=args.feature_sets,
            model_types=args.models,
            use_grid_search=not args.no_grid_search
        )
    
    print("\n" + "="*70)
    print("HOAN THANH!")
    print("="*70)


if __name__ == '__main__':
    main()
