import pandas as pd

# Đọc file
df = pd.read_csv('data/filtered_images_index.csv')

# Xem các lớp theo từng crop
for crop in df['crop'].unique():
    classes = df[df['crop'] == crop]['disease'].unique()
    print(f"\n{crop}:")
    for c in sorted(classes):
        print(f"  - {c}")