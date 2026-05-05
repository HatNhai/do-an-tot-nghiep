"""
Module trích xuất feature cho ảnh lá chè bệnh.
Bao gồm 4 bộ feature: F1 (Color), F2 (Color+Veg), F3 (Color+Texture), F4 (Full)
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2hsv, rgb2lab


def read_image(image_path):
    """
    Đọc ảnh và convert sang RGB.
    
    Args:
        image_path: Đường dẫn đến file ảnh
    
    Returns:
        numpy array RGB image (H, W, 3)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    # OpenCV đọc BGR, convert sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def extract_color_features(img):
    """
    Trích xuất feature nhóm F1 - Màu cơ bản.
    
    Bao gồm:
    - RGB mean & std (6 features)
    - HSV mean & std (6 features)
    
    Args:
        img: Ảnh RGB (H, W, 3)
    
    Returns:
        dict với keys: f1_features (numpy array)
    """
    # RGB features
    mean_rgb = np.mean(img, axis=(0, 1))
    std_rgb = np.std(img, axis=(0, 1))
    
    # HSV features
    img_hsv = rgb2hsv(img)
    mean_hsv = np.mean(img_hsv, axis=(0, 1))
    std_hsv = np.std(img_hsv, axis=(0, 1))
    
    # Kết hợp: R_mean, G_mean, B_mean, R_std, G_std, B_std,
    #          H_mean, S_mean, V_mean, H_std, S_std, V_std
    f1_features = np.concatenate([
        mean_rgb, std_rgb,
        mean_hsv, std_hsv
    ])
    
    return {
        'f1_features': f1_features,
        'f1_names': [
            'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std',
            'H_mean', 'S_mean', 'V_mean', 'H_std', 'S_std', 'V_std'
        ]
    }


def extract_vegetation_features(img):
    """
    Trích xuất feature nhóm F2 - Chỉ số vegetation-like từ RGB.
    
    Bao gồm:
    - VARI (Visible Atmospherically Resistant Index)
    - Normalized RGB channels
    
    Args:
        img: Ảnh RGB (H, W, 3)
    
    Returns:
        dict với keys: f2_features (numpy array)
    """
    R = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    B = img[:, :, 2].astype(np.float32)
    
    # VARI = (G - R) / (G + R - B)
    epsilon = 1e-6
    denominator = G + R - B + epsilon
    VARI = (G - R) / denominator
    
    # Tính statistics cho VARI
    mean_vari = np.mean(VARI)
    std_vari = np.std(VARI)
    p10_vari = np.percentile(VARI, 10)
    p90_vari = np.percentile(VARI, 90)
    
    # Normalized RGB
    sum_rgb = R + G + B + epsilon
    R_norm = np.mean(R / sum_rgb)
    G_norm = np.mean(G / sum_rgb)
    B_norm = np.mean(B / sum_rgb)
    
    f2_features = np.array([
        mean_vari, std_vari, p10_vari, p90_vari,
        R_norm, G_norm, B_norm
    ])
    
    return {
        'f2_features': f2_features,
        'f2_names': [
            'VARI_mean', 'VARI_std', 'VARI_p10', 'VARI_p90',
            'R_norm', 'G_norm', 'B_norm'
        ]
    }


def extract_texture_features(img, reduce_levels=16):
    """
    Trích xuất feature nhóm F3 - Texture (GLCM + LBP).
    
    Bao gồm:
    - GLCM: contrast, homogeneity, energy, correlation (4 features)
    - LBP histogram (59 bins cho uniform pattern)
    
    Args:
        img: Ảnh RGB (H, W, 3)
        reduce_levels: Số mức xám để giảm (mặc định 16)
    
    Returns:
        dict với keys: f3_features (numpy array)
    """
    # Convert sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Giảm mức xám để GLCM ổn định
    gray_reduced = (gray // (256 // reduce_levels)).astype(np.uint8)
    
    # GLCM features
    # Tính GLCM với distance=1 và các hướng: 0°, 45°, 90°, 135°
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(
        gray_reduced,
        distances=distances,
        angles=angles,
        levels=reduce_levels,
        symmetric=True,
        normed=True
    )
    
    # Tính các thuộc tính GLCM
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    glcm_features = np.array([contrast, homogeneity, energy, correlation])
    
    # LBP features
    # Uniform LBP với P=8, R=1
    P = 8  # Số điểm xung quanh
    R = 1  # Bán kính
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    
    # Histogram LBP (uniform pattern có P+2 bins = 10 bins)
    # Nhưng để đầy đủ hơn, dùng 59 bins (P*(P-1)+3 cho uniform)
    n_bins = P * (P - 1) + 3  # 59 bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # Kết hợp GLCM + LBP
    f3_features = np.concatenate([glcm_features, hist])
    
    return {
        'f3_features': f3_features,
        'f3_names': (
            ['GLCM_contrast', 'GLCM_homogeneity', 'GLCM_energy', 'GLCM_correlation'] +
            [f'LBP_bin_{i}' for i in range(n_bins)]
        )
    }


def extract_all_features(image_path, resize_size=None):
    """
    Trích xuất tất cả features (F1-F4) cho một ảnh.
    
    Args:
        image_path: Đường dẫn đến file ảnh
        resize_size: Tuple (H, W) để resize ảnh (None = giữ nguyên)
    
    Returns:
        dict chứa:
        - f1_features: Color features
        - f2_features: Color + Vegetation features
        - f3_features: Color + Texture features
        - f4_features: Full features (F1 + F2 + F3)
        - feature_names: Tên các feature
    """
    # Đọc ảnh
    img = read_image(image_path)
    
    # Resize nếu cần
    if resize_size is not None:
        img = cv2.resize(img, resize_size[::-1])  # cv2.resize dùng (W, H)
    
    # Extract các nhóm feature
    f1_dict = extract_color_features(img)
    f2_dict = extract_vegetation_features(img)
    f3_dict = extract_texture_features(img)
    
    # Tổng hợp
    f1_features = f1_dict['f1_features']
    f2_features = f2_dict['f2_features']
    f3_features = f3_dict['f3_features']
    
    # F2 = F1 + Vegetation
    f2_full = np.concatenate([f1_features, f2_features])
    
    # F3 = F1 + Texture
    f3_full = np.concatenate([f1_features, f3_features])
    
    # F4 = F1 + F2 + F3 (full)
    f4_features = np.concatenate([f1_features, f2_features, f3_features])
    
    # Tên feature
    f1_names = f1_dict['f1_names']
    f2_names = f1_names + f2_dict['f2_names']
    f3_names = f1_names + f3_dict['f3_names']
    f4_names = f1_names + f2_dict['f2_names'] + f3_dict['f3_names']
    
    return {
        'f1_features': f1_features,
        'f2_features': f2_full,
        'f3_features': f3_full,
        'f4_features': f4_features,
        'f1_names': f1_names,
        'f2_names': f2_names,
        'f3_names': f3_names,
        'f4_names': f4_names
    }


if __name__ == '__main__':
    # Test script
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Tìm một ảnh test
        test_dirs = ['Healthy', 'TEALEA~1', 'TEARED~1', 'TEARED~2']
        test_image = None
        for d in test_dirs:
            p = Path(d)
            if p.exists():
                imgs = list(p.glob('*.jpg'))
                if imgs:
                    test_image = imgs[0]
                    break
        
        if test_image is None:
            print("Không tìm thấy ảnh test. Usage: python feature_extraction.py <image_path>")
            sys.exit(1)
    
    print(f"Đang test với ảnh: {test_image}")
    features = extract_all_features(test_image)
    
    print(f"\nF1 (Color) features: {len(features['f1_features'])}")
    print(f"F2 (Color+Veg) features: {len(features['f2_features'])}")
    print(f"F3 (Color+Texture) features: {len(features['f3_features'])}")
    print(f"F4 (Full) features: {len(features['f4_features'])}")
    
    print("\nMột số giá trị F1:", features['f1_features'][:6])
    print("Một số giá trị F2:", features['f2_features'][:6])
    print("Một số giá trị F3:", features['f3_features'][:6])
    print("Một số giá trị F4:", features['f4_features'][:6])
