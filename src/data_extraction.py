import os
import shutil
import math
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits dataset into train/val/test folders from class folders inside dataset/.
    """

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    random.seed(42)

    classes = [d for d in source_dir.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes to split.\n")

    for cls_dir in tqdm(classes, desc="Splitting classes"):
        images = [img for img in cls_dir.iterdir() if img.is_file()]
        
        
        random.shuffle(images)

        n_total = len(images)
        if n_total < 10:
            print(f"Skipping {cls_dir.name}, not enough images ({n_total})")
            continue

        n_train = math.floor(n_total * train_ratio)
        n_val = math.floor(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs   = images[n_train:n_train + n_val]
        test_imgs  = images[n_train + n_val:]

       
        for split in ["train", "val", "test"]:
            (output_dir / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

       
        for img in train_imgs:
            shutil.copy(img, output_dir / "train" / cls_dir.name / img.name)

        for img in val_imgs:
            shutil.copy(img, output_dir / "val" / cls_dir.name / img.name)

        for img in test_imgs:
            shutil.copy(img, output_dir / "test" / cls_dir.name / img.name)

        print(f"{cls_dir.name}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    print("\nDataset split completed.")
    print(f"Train={train_ratio*100}%, Val={val_ratio*100}%, Test={test_ratio*100}%")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent  
    SOURCE = ROOT / "dataset"          
    OUTPUT = ROOT / "dataset_split"   

    split_dataset(SOURCE, OUTPUT)
