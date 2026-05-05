"""
Script 02: Gán severity labels tự động dựa trên segmentation
Severity levels: 0=healthy, 1=mild (<15%), 2=moderate (15-40%), 3=severe (>40%)
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_severity(image_path, label_name):
    """
    Tính severity level và ratio dựa trên segmentation.
    
    Returns:
        severity_level: 0, 1, 2, hoặc 3
        severity_ratio: Tỉ lệ vùng bệnh (0.0 - 1.0)
        disease_mask: Mask vùng bệnh (để visualize)
    """
    # 1. Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Segment healthy leaf (green)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    
    # 3. Remove background
    # Background = very dark (V<30) or very bright (V>240)
    v_channel = img_hsv[:, :, 2]
    background_mask = (v_channel < 30) | (v_channel > 240)
    
    # 4. Leaf mask = not background
    leaf_mask = ~background_mask
    
    # 5. Disease mask = leaf but not green
    disease_mask = leaf_mask & (~green_mask)
    
    # 6. Apply morphological operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disease_mask_uint8 = disease_mask.astype(np.uint8) * 255
    disease_mask_cleaned = cv2.morphologyEx(disease_mask_uint8,
                                           cv2.MORPH_OPEN, kernel)
    disease_mask_cleaned = disease_mask_cleaned > 0
    
    # 7. Compute ratio
    total_leaf = np.sum(leaf_mask)
    disease_area = np.sum(disease_mask_cleaned & leaf_mask)
    severity_ratio = disease_area / (total_leaf + 1e-6)
    
    # 8. Map to severity level
    if "healthy" in label_name.lower():
        severity_level = 0
    elif severity_ratio < 0.15:
        severity_level = 1
    elif severity_ratio < 0.40:
        severity_level = 2
    else:
        severity_level = 3
    
    return severity_level, severity_ratio, disease_mask_cleaned


def label_severity(index_file='data/filtered_images_index.csv',
                  output_file='data/severity_labels.csv',
                  visualize_samples=True,
                  n_samples_per_severity=5):
    """
    Gán severity labels cho tất cả ảnh.
    
    Args:
        index_file: File CSV chứa index ảnh
        output_file: File CSV để lưu kết quả
        visualize_samples: Có visualize mẫu để kiểm tra không
        n_samples_per_severity: Số mẫu visualize mỗi severity level
    """
    # Đọc index
    df = pd.read_csv(index_file)
    
    print(f"Dang gan severity labels cho {len(df)} anh...")
    
    results = []
    failed_images = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing severity"):
        try:
            severity_level, severity_ratio, disease_mask = compute_severity(
                row['image_path'],
                row['original_label']
            )
            
            if severity_level is None:
                failed_images.append(row['image_path'])
                continue
            
            results.append({
                'image_path': row['image_path'],
                'crop': row['crop'],
                'disease': row['disease'],
                'severity': severity_level,
                'severity_ratio': severity_ratio,
                'split': row['split']
            })
            
        except Exception as e:
            print(f"\nError processing {row['image_path']}: {e}")
            failed_images.append(row['image_path'])
            continue
    
    # Tạo DataFrame
    df_severity = pd.DataFrame(results)
    
    if len(df_severity) == 0:
        print("ERROR: Khong tinh duoc severity cho anh nao!")
        return None
    
    # Lưu kết quả
    df_severity.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("KET QUA SEVERITY LABELING")
    print(f"{'='*70}")
    print(f"Tong so anh: {len(df_severity)}")
    print(f"So anh loi: {len(failed_images)}")
    
    print(f"\nPhan bo theo severity:")
    print(df_severity['severity'].value_counts().sort_index())
    
    print(f"\nPhan bo theo crop x severity:")
    print(df_severity.groupby(['crop', 'severity']).size())
    
    print(f"\nPhan bo theo disease x severity:")
    print(df_severity.groupby(['disease', 'severity']).size())
    
    print(f"\nThong ke severity_ratio:")
    print(df_severity['severity_ratio'].describe())
    
    print(f"\nDa luu ket qua vao: {output_file}")
    
    # Visualize samples nếu cần
    if visualize_samples:
        visualize_severity_samples(df_severity, n_samples_per_severity)
    
    return df_severity


def visualize_severity_samples(df_severity, n_samples=5):
    """
    Visualize một số mẫu để kiểm tra severity labeling.
    """
    output_dir = Path('results/severity_samples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDang visualize {n_samples} mau moi severity level...")
    
    for severity in [0, 1, 2, 3]:
        df_sev = df_severity[df_severity['severity'] == severity]
        if len(df_sev) == 0:
            continue
        
        # Lấy n_samples ngẫu nhiên
        samples = df_sev.sample(min(n_samples, len(df_sev)), random_state=42)
        
        fig, axes = plt.subplots(1, len(samples), figsize=(4*len(samples), 4))
        if len(samples) == 1:
            axes = [axes]
        
        for i, (_, row) in enumerate(samples.iterrows()):
            img = cv2.imread(row['image_path'])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"Severity {severity}\nRatio: {row['severity_ratio']:.3f}\n{row['disease']}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'severity_{severity}_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Da luu visualization vao: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Gan severity labels')
    parser.add_argument('--index', type=str, default='data/filtered_images_index.csv',
                       help='File CSV chua index anh')
    parser.add_argument('--output', type=str, default='data/severity_labels.csv',
                       help='File CSV de luu ket qua')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Khong visualize samples')
    parser.add_argument('--sample-size', type=int, default=5,
                       help='So mau visualize moi severity level')
    
    args = parser.parse_args()
    
    df_severity = label_severity(
        args.index,
        args.output,
        visualize_samples=not args.no_visualize,
        n_samples_per_severity=args.sample_size
    )
