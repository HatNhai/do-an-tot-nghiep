import os
import pandas as pd

DATA_DIR = "data/raw"

# Mapping nhãn
CLASS_MAP = {
    "Healthy": 0,
    "TeaLeafBlight": 1,
    "TeaRedLeafSpot": 2,
    "TeaRedScab": 3
}

rows = []

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    if class_name not in CLASS_MAP:
        print(f"Bỏ qua thư mục không hợp lệ: {class_name}")
        continue

    label = CLASS_MAP[class_name]

    for img_name in os.listdir(class_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            rel_path = os.path.join("data/raw", class_name, img_name)
            rows.append([rel_path, class_name, label])

df = pd.DataFrame(rows, columns=["image_path", "class_name", "label"])
df.to_csv("image_index.csv", index=False)

print("Đã tạo image_index.csv")
print(df["class_name"].value_counts())