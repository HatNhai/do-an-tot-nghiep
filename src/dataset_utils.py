"""
Module để chuẩn hóa và index dữ liệu ảnh lá chè.
Tạo DataFrame chứa đường dẫn ảnh và nhãn lớp.
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# Mapping tên lớp sang label
CLASS_MAPPING = {
    'Healthy': 0,
    'Leaf_Blight': 1,
    'Tea_Red_Leaf_Spot': 2,
    'Tea_Red_Scab': 3
}

# Mapping ngược từ label sang tên lớp
LABEL_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}


def build_image_index(data_dir, class_mapping=None):
    """
    Duyệt qua các thư mục con trong data_dir và tạo index DataFrame.
    
    Args:
        data_dir: Đường dẫn thư mục chứa các lớp (ví dụ: 'data/raw')
        class_mapping: Dict mapping tên lớp -> label (mặc định dùng CLASS_MAPPING)
    
    Returns:
        DataFrame với columns: image_path, class_name, label
    """
    if class_mapping is None:
        class_mapping = CLASS_MAPPING
    
    data_dir = Path(data_dir)
    records = []
    
    # Duyệt qua các thư mục con
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Chuẩn hóa tên lớp (xử lý các biến thể tên)
        class_name_normalized = normalize_class_name(class_name)
        
        # Bỏ qua nếu không có trong mapping
        if class_name_normalized not in class_mapping:
            print(f"Warning: Bỏ qua thư mục '{class_name}' (không có trong class_mapping)")
            continue
        
        label = class_mapping[class_name_normalized]
        
        # Duyệt qua các file ảnh
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                records.append({
                    'image_path': str(img_file.absolute()),
                    'class_name': class_name_normalized,
                    'label': label
                })
    
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        raise ValueError(f"Không tìm thấy ảnh nào trong {data_dir}")
    
    print(f"Đã index {len(df)} ảnh từ {data_dir}")
    print("\nPhân bố theo lớp:")
    print(df['class_name'].value_counts().sort_index())
    
    return df


def normalize_class_name(class_name):
    """
    Chuẩn hóa tên lớp từ tên thư mục.
    Xử lý các biến thể như 'TEALEA~1', 'TEARED~1', v.v.
    """
    class_name_upper = class_name.upper()
    
    # Mapping các biến thể tên thư mục
    name_mapping = {
        'HEALTHY': 'Healthy',
        'TEALEA': 'Leaf_Blight',
        'TEALEA~1': 'Leaf_Blight',
        'LEAF_BLIGHT': 'Leaf_Blight',
        'LEAFBLIGHT': 'Leaf_Blight',
        'TEARED': 'Tea_Red_Leaf_Spot',  # Cần kiểm tra thứ tự
        'TEARED~1': 'Tea_Red_Leaf_Spot',
        'TEARED~2': 'Tea_Red_Scab',
        'TEA_RED_LEAF_SPOT': 'Tea_Red_Leaf_Spot',
        'TEA_RED_SCAB': 'Tea_Red_Scab',
        'TEAREDLEAFSPOT': 'Tea_Red_Leaf_Spot',
        'TEAREDSCAB': 'Tea_Red_Scab'
    }
    
    # Tìm match
    for key, normalized in name_mapping.items():
        if key in class_name_upper or class_name_upper.startswith(key.replace('~', '')):
            return normalized
    
    # Nếu không match, trả về tên gốc với format chuẩn
    return class_name.replace(' ', '_').replace('-', '_')


def split_dataset(index_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, 
                  random_state=42, stratify_col='label'):
    """
    Chia dataset thành train/val/test với tỉ lệ cân bằng lớp.
    
    Args:
        index_df: DataFrame chứa image_path, label
        train_ratio: Tỉ lệ train (mặc định 0.6)
        val_ratio: Tỉ lệ validation (mặc định 0.2)
        test_ratio: Tỉ lệ test (mặc định 0.2)
        random_state: Random seed
        stratify_col: Cột để stratify (mặc định 'label')
    
    Returns:
        DataFrame với cột 'split' thêm vào
    """
    # Kiểm tra tỉ lệ
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Tổng tỉ lệ phải bằng 1.0, nhận được {total_ratio}")
    
    df = index_df.copy()
    
    # Chia train và temp (val + test)
    X = df.drop(columns=[stratify_col] if stratify_col in df.columns else [])
    y = df[stratify_col]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=y
    )
    
    # Chia temp thành val và test
    val_size_in_temp = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size_in_temp),
        random_state=random_state,
        stratify=y_temp
    )
    
    # Gán split
    df.loc[X_train.index, 'split'] = 'train'
    df.loc[X_val.index, 'split'] = 'val'
    df.loc[X_test.index, 'split'] = 'test'
    
    print("\nPhân bố train/val/test:")
    print(df.groupby(['split', 'class_name']).size().unstack(fill_value=0))
    print(f"\nTổng số ảnh: {len(df)}")
    print(f"Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Val: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Test: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
    
    return df


if __name__ == '__main__':
    # Test script
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = '.'  # Thư mục hiện tại
    
    print(f"Đang index dữ liệu từ: {data_dir}")
    df = build_image_index(data_dir)
    
    print("\nĐang chia train/val/test...")
    df_split = split_dataset(df)
    
    # Lưu kết quả
    output_file = 'data/images_index.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_split.to_csv(output_file, index=False)
    print(f"\nĐã lưu index vào: {output_file}")
