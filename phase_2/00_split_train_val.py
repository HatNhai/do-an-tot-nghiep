import os
import random
import shutil
from pathlib import Path

# Mapping rename class cho đúng format script 01
CLASS_MAPPING = {
    "Tomato___Early_blight": "Tomato__Tomato_Early_blight",
    "Tomato___Late_blight": "Tomato__Tomato_Late_blight",
    "Tomato___Septoria_leaf_spot": "Tomato__Tomato_Septoria_leaf_spot",
    "Tomato___healthy": "Tomato__Tomato_Healthy",

    "Potato___Early_blight": "Potato__Potato_Early_blight",
    "Potato___Late_blight": "Potato__Potato_Late_blight",
    "Potato___healthy": "Potato__Potato_Healthy",
}


def split_dataset(source_dir="data/raw", output_dir="archive", train_ratio=0.8):
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    train_path = output_path / "train"
    val_path = output_path / "val"

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    print("Bắt đầu split dataset...\n")

    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue

        original_name = class_dir.name

        # Bỏ qua class không dùng
        if original_name not in CLASS_MAPPING:
            print(f"Skip: {original_name}")
            continue

        new_name = CLASS_MAPPING[original_name]

        images = list(class_dir.glob("*"))
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Tạo thư mục mới
        (train_path / new_name).mkdir(parents=True, exist_ok=True)
        (val_path / new_name).mkdir(parents=True, exist_ok=True)

        # Copy train
        for img in train_imgs:
            shutil.copy2(img, train_path / new_name / img.name)

        # Copy val
        for img in val_imgs:
            shutil.copy2(img, val_path / new_name / img.name)

        print(f"{new_name}: train={len(train_imgs)}, val={len(val_imgs)}")

    print("\nDone! Dataset đã sẵn sàng cho Script 01")


if __name__ == "__main__":
    split_dataset()