import pandas as pd
from pathlib import Path

# Đọc file
df = pd.read_csv('data/severity_labels.csv')

def convert_path(p):
    p = str(p)

    # Tìm phần từ "data/filtered"
    idx = p.lower().find("data\\filtered")
    if idx == -1:
        idx = p.lower().find("data/filtered")

    if idx != -1:
        new_path = p[idx:].replace("\\", "/")
        return new_path

    return p  # fallback nếu không match

# Apply
df['image_path'] = df['image_path'].apply(convert_path)

# Lưu lại
df.to_csv('data/severity_labels.csv', index=False)

print("Đã convert path xong!")