import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================
# PATHS
# ======================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset_split"
SAVE_DIR = ROOT / "models"
SAVE_DIR.mkdir(exist_ok=True)

# ======================================
# CONFIG
# ======================================
BATCH_SIZE = 16   # smaller batch helps CPU
LR = 1e-4
EPOCHS = 20
PATIENCE = 5
DEVICE = "cpu"     # force CPU

print("Using Device:", DEVICE)

# ======================================
# TRANSFORMS (CPU-friendly)
# ======================================
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# ======================================
# DATASETS + LOADERS
# ======================================
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_dataset   = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))

# ======================================
# MODEL
# ======================================
num_classes = len(train_dataset.classes)

model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

# ======================================
# TRAINING LOOP
# ======================================
best_val_loss = float("inf")
epochs_no_improve = 0
train_losses = []
val_losses = []
stop_epoch = None

print("\nStarting Training...\n")

for epoch in range(1, EPOCHS + 1):

    # ---------------- TRAIN -----------------
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # ---------------- VALIDATE -----------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    print(f"Epoch {epoch}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # ------- EARLY STOPPING -------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), SAVE_DIR / "best_model_cpu.pt")
        print("→ Model improved. Saved.")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        stop_epoch = epoch
        break

if stop_epoch is None:
    stop_epoch = EPOCHS

# ======================================
# PLOT RESULTS
# ======================================
plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Val Loss", marker="o")
plt.axvline(stop_epoch-1, color="red", linestyle="--", label=f"Stopped @ {stop_epoch}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss (CPU Training)")
plt.legend()
plt.grid(True)

plt.savefig(ROOT / "training_curve_cpu.png")
print("\nSaved training curve as training_curve_cpu.png")

plt.show()
