import os
import shutil
import random
from pathlib import Path

# Путь к вашей директории
BASE_DIR = Path(r"D:\Файлы и всё такое\Институт\Практика 3 год\data")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "val"

# Классы
CLASSES = ["real", "fake", "notdoc"]

# Параметр разбиения
val_ratio = 0.2  # 20% в валидацию

for cls in CLASSES:
    train_cls_dir = TRAIN_DIR / cls
    val_cls_dir = VAL_DIR / cls

    # Создаем папку val/class, если её нет
    val_cls_dir.mkdir(parents=True, exist_ok=True)

    # Получаем список всех .jpg файлов в папке train/class
    images = [f for f in os.listdir(train_cls_dir) if f.lower().endswith(".jpg")]
    random.shuffle(images)

    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]

    print(f"{cls}: Перемещаем {val_count} файлов из {len(images)} в валидацию")

    for fname in val_images:
        src = train_cls_dir / fname
        dst = val_cls_dir / fname
        shutil.move(str(src), str(dst))
