"""
Script chạy pipeline với 50% dataset để tạo báo cáo draft.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Chạy command và hiển thị progress."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\nError: {description} failed!")
        return False
    return True

def main():
    # Tính 50% dataset
    import pandas as pd
    index_file = 'images_index_split.csv'
    if not os.path.exists(index_file):
        print("Chua co index file. Dang chay buoc 1...")
        if not run_command('python main.py --step 1 --data-dir .', 'Buoc 1: Index du lieu'):
            return
    
    df = pd.read_csv(index_file)
    total = len(df)
    sample_size = int(total * 0.5)
    
    print(f"\n{'='*70}")
    print(f"CHAY PIPELINE VOI 50% DATASET")
    print(f"{'='*70}")
    print(f"Tong so anh: {total}")
    print(f"50% sample: {sample_size}")
    print(f"{'='*70}\n")
    
    # Bước 2: Extract features với 50% data
    print("\nDang chay buoc 2: Trich xuat features (50% dataset)...")
    cmd = f'python main.py --step 2 --sample-size {sample_size}'
    if not run_command(cmd, 'Buoc 2: Extract features'):
        print("Loi o buoc 2!")
        return
    
    # Bước 3: Train và evaluate với tất cả feature sets và models
    print("\nDang chay buoc 3: Train va evaluate...")
    cmd = 'python main.py --step 3 --feature-sets f1 f2 f3 f4 --models svm rf --no-grid-search'
    if not run_command(cmd, 'Buoc 3: Train & Evaluate'):
        print("Loi o buoc 3!")
        return
    
    # Tạo report
    print("\nDang tao bao cao...")
    cmd = 'python generate_report.py'
    if not run_command(cmd, 'Tao bao cao'):
        print("Loi khi tao bao cao!")
        return
    
    print(f"\n{'='*70}")
    print("HOAN THANH!")
    print(f"{'='*70}")
    print(f"Ket qua da duoc luu vao:")
    print(f"  - features/results_summary.csv")
    print(f"  - report.md")
    print(f"  - report.html (neu co)")
    print(f"\nDe xem ket qua, mo file report.md hoac report.html")

if __name__ == '__main__':
    main()
