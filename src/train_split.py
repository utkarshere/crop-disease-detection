import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SRC_TRAIN = ROOT / "PlantDoc_Dataset" / "train"
SRC_TEST  = ROOT / "PlantDoc_Dataset" / "test"

DEST = ROOT / "dataset_split"
TRAIN_OUT = DEST / "train"
VAL_OUT   = DEST / "val"
TEST_OUT  = DEST / "test"

def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    TRAIN_OUT.mkdir(parents=True)
    VAL_OUT.mkdir(parents=True)
    TEST_OUT.mkdir(parents=True)

    print("Splitting train → train + val\n")
    for class_dir in SRC_TRAIN.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*"))
        class_name = class_dir.name

        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        (TRAIN_OUT / class_name).mkdir(parents=True)
        (VAL_OUT / class_name).mkdir(parents=True)

        for img in train_imgs:
            shutil.copy(img, TRAIN_OUT / class_name / img.name)

        for img in val_imgs:
            shutil.copy(img, VAL_OUT / class_name / img.name)

        print(f"{class_name}: Train={len(train_imgs)}, Val={len(val_imgs)}")

    print("\nCopying test set...")
    for class_dir in SRC_TEST.iterdir():
        if not class_dir.is_dir():
            continue

        (TEST_OUT / class_dir.name).mkdir(exist_ok=True)
        for img in class_dir.glob("*"):
            shutil.copy(img, TEST_OUT / class_dir.name / img.name)

    print("\nDone! dataset_split/ created.")

if __name__ == "__main__":
    main()
