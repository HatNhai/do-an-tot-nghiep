import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc file index
df = pd.read_csv("image_index.csv")

# ======================
# Bước 1: 60% train
# ======================
train_df, temp_df = train_test_split(
    df,
    test_size=0.4,          # 40% còn lại
    stratify=df["label"],   # QUAN TRỌNG
    random_state=42
)

# ======================
# Bước 2: 20% val / 20% test
# ======================
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

# ======================
# Thêm cột split
# ======================
train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

final_df = pd.concat([train_df, val_df, test_df])

# Lưu lại
final_df.to_csv("image_index_split.csv", index=False)

print("Đã chia xong!")
print(final_df.groupby(["split", "class_name"]).size())

print("\nTỉ lệ phần trăm mỗi lớp trong từng split:\n")

distribution = (
    final_df
    .groupby("split")["class_name"]
    .value_counts(normalize=True)
    .mul(100)
)

print(distribution)