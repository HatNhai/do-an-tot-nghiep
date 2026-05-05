"""
Script 03: Trích xuất features cho tất cả ảnh
Tái sử dụng pipeline Phase 1 với điều chỉnh cho Phase 2
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
from pathlib import Path

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_extraction import extract_all_features


def extract_features_for_dataset(severity_file='data/severity_labels.csv',
                                output_file='data/leaf_features_with_severity.csv',
                                resize_size=None,
                                sample_size=None):
    """
    Trích xuất features cho tất cả ảnh trong dataset.
    
    Args:
        severity_file: File CSV chứa severity labels
        output_file: File CSV để lưu features
        resize_size: Tuple (H, W) để resize ảnh
        sample_size: Số ảnh để test (None = tất cả)
    """
    # Đọc severity labels
    df_severity = pd.read_csv(severity_file)
    
    # Sample nếu cần
    if sample_size and sample_size < len(df_severity):
        print(f"Warning: Chi xu ly {sample_size} anh (stratified sample)")
        from sklearn.model_selection import train_test_split
        df_severity, _ = train_test_split(
            df_severity,
            train_size=sample_size,
            stratify=df_severity[['severity', 'crop', 'disease']].apply(
                lambda x: f"{x['severity']}_{x['crop']}_{x['disease']}", axis=1
            ),
            random_state=42
        )
    
    print(f"\nDang trich xuat features cho {len(df_severity)} anh...")
    
    all_features = []
    failed_images = []
    
    for idx, row in tqdm(df_severity.iterrows(), total=len(df_severity), desc="Extracting"):
        try:
            features_dict = extract_all_features(row['image_path'], resize_size=resize_size)
            
            # Tạo record
            record = {
                'image_path': row['image_path'],
                'crop': row['crop'],
                'disease': row['disease'],
                'severity': row['severity'],
                'severity_ratio': row['severity_ratio'],
                'split': row['split']
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
    
    print(f"\n{'='*70}")
    print("KET QUA TRICH XUAT FEATURES")
    print(f"{'='*70}")
    print(f"Tong so anh: {len(df_features)}")
    print(f"So feature F1: {len([c for c in df_features.columns if c.startswith('f1_')])}")
    print(f"So feature F2: {len([c for c in df_features.columns if c.startswith('f2_')])}")
    print(f"So feature F3: {len([c for c in df_features.columns if c.startswith('f3_')])}")
    
    print(f"\nDa luu features vao: {output_file}")
    
    return df_features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trich xuat features')
    parser.add_argument('--severity-file', type=str, default='data/severity_labels.csv',
                       help='File CSV chua severity labels')
    parser.add_argument('--output', type=str, default='data/leaf_features_with_severity.csv',
                       help='File CSV de luu features')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('H', 'W'),
                       help='Resize anh ve kich thuoc HxW')
    parser.add_argument('--sample-size', type=int,
                       help='So anh de test (None = tat ca)')
    
    args = parser.parse_args()
    
    resize_size = tuple(args.resize) if args.resize else None
    
    df_features = extract_features_for_dataset(
        args.severity_file,
        args.output,
        resize_size=resize_size,
        sample_size=args.sample_size
    )
