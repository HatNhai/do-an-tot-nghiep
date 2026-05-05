"""
Script 01: Lọc dataset cho Nhánh 06
Chọn subset: Tomato (Early_blight, Late_blight, Septoria_leaf_spot, healthy) 
và Potato (Early_blight, Late_blight, healthy)
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# Định nghĩa các lớp cần lọc
SELECTED_CLASSES = {
    'Tomato': [
        'Tomato__Tomato_Early_blight',
        'Tomato__Tomato_Late_blight',
        'Tomato__Tomato_Septoria_leaf_spot',
        'Tomato__Tomato_Healthy'
    ],
    'Potato': [
        'Potato__Potato_Early_blight',
        'Potato__Potato_Late_blight',
        'Potato__Potato_Healthy'
    ]
}


def parse_class_name(class_name):
    """
    Parse class name để lấy crop và disease.
    Ví dụ: 'Tomato__Tomato_Early_blight' -> ('Tomato', 'Early_blight')
    """
    parts = class_name.split('__')
    if len(parts) < 2:
        return None, None
    
    crop = parts[0]
    disease_part = parts[1]
    
    # Loại bỏ prefix crop nếu có
    if disease_part.startswith(crop + '_'):
        disease = disease_part[len(crop) + 1:]
    else:
        disease = disease_part
    
    # Chuẩn hóa disease name
    if 'healthy' in disease.lower():
        disease = 'healthy'
    
    return crop, disease


def filter_dataset(source_dir='archive', output_dir='data/filtered', splits=['train', 'val']):
    """
    Lọc dataset và tạo index file.
    
    Args:
        source_dir: Thư mục chứa dataset gốc
        output_dir: Thư mục để lưu các lớp đã lọc
        splits: Các split cần xử lý ['train', 'val', 'test']
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    print("Dang loc dataset...")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    
    # Duyệt qua các split
    for split in splits:
        split_source = source_path / split
        if not split_source.exists():
            print(f"Warning: Khong tim thay {split_source}, bo qua")
            continue
        
        print(f"\nXu ly split: {split}")
        
        # Duyệt qua các thư mục lớp
        for class_dir in tqdm(sorted(split_source.iterdir()), desc=f"Processing {split}"):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Kiểm tra xem có trong danh sách chọn không
            crop, disease = parse_class_name(class_name)
            
            if crop is None or disease is None:
                continue
            
            # Kiểm tra xem có trong SELECTED_CLASSES không
            if crop not in SELECTED_CLASSES:
                continue
            
            if class_name not in SELECTED_CLASSES[crop]:
                continue
            
            # Tạo thư mục đích
            target_dir = output_path / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Đếm và copy ảnh
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_count = 0
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    # Copy file
                    target_file = target_dir / img_file.name
                    if not target_file.exists():
                        shutil.copy2(img_file, target_file)
                    
                    # Ghi record
                    records.append({
                        'image_path': str(target_file.absolute()),
                        'crop': crop,
                        'disease': disease,
                        'original_label': class_name,
                        'split': split
                    })
                    image_count += 1
            
            print(f"  {class_name}: {image_count} anh")
    
    # Tạo DataFrame và lưu
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        print("ERROR: Khong tim thay anh nao!")
        return None
    
    output_file = 'data/filtered_images_index.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("KET QUA LOC DATASET")
    print(f"{'='*70}")
    print(f"Tong so anh: {len(df)}")
    print(f"\nPhan bo theo crop:")
    print(df['crop'].value_counts())
    print(f"\nPhan bo theo disease:")
    print(df['disease'].value_counts())
    print(f"\nPhan bo theo split:")
    print(df['split'].value_counts())
    print(f"\nPhan bo theo crop x disease:")
    print(df.groupby(['crop', 'disease']).size())
    
    print(f"\nDa luu index vao: {output_file}")
    
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Loc dataset cho Nhanh 06')
    parser.add_argument('--source', type=str, default='archive',
                       help='Thu muc chua dataset goc')
    parser.add_argument('--output', type=str, default='data/filtered',
                       help='Thu muc de luu cac lop da loc')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                       help='Cac split can xu ly')
    
    args = parser.parse_args()
    
    df = filter_dataset(args.source, args.output, args.splits)
