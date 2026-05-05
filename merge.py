import pandas as pd

# Đọc file
df_features = pd.read_csv('features/leaf_features.csv')
df_split = pd.read_csv('image_index_split.csv')  # file có split đúng

# Giữ lại cột cần thiết bên split
df_split = df_split[['image_path', 'split']]

# Merge theo image_path
df_merged = df_features.merge(df_split, on='image_path', how='left', suffixes=('', '_new'))

# Ghi đè cột split cũ
df_merged['split'] = df_merged['split_new']

# Xoá cột thừa
df_merged.drop(columns=['split_new'], inplace=True)

# Check lại
print(df_merged['split'].value_counts())

# Lưu lại
df_merged.to_csv('features/leaf_features.csv', index=False)

print("Merge xong, ready train")