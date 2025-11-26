import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path
import numpy as np
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "dataset_split" / "test"
MODEL_PATH = ROOT / "models" / "best_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {DEVICE}\n")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
num_classes = len(checkpoint['classes'])
class_names = checkpoint['classes']

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes) #type: ignore
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

print(f"Model loaded from: {MODEL_PATH}")
print(f"Training Val Accuracy: {checkpoint['val_acc']:.2f}%\n")
print("="*80)

correct = 0
total = 0
confidence_scores = []
correct_confidences = []
incorrect_confidences = []
per_class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})

print(f"{'Image':<40} {'True Label':<35} {'Predicted':<35} {'Confidence':<12} {'Status'}")
print("="*80)

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
        
        confidence_val = confidence.item() * 100
        predicted_class = class_names[predicted.item()]
        true_class = class_names[labels.item()]
        
        is_correct = predicted.eq(labels).item()
        total += 1
        correct += is_correct
        
        confidence_scores.append(confidence_val)
        per_class_stats[true_class]['total'] += 1 #type: ignore
        per_class_stats[true_class]['confidences'].append(confidence_val) #type: ignore
        
        if is_correct:
            correct_confidences.append(confidence_val)
            per_class_stats[true_class]['correct'] += 1 #type: ignore
            status = "✓ CORRECT"
        else:
            incorrect_confidences.append(confidence_val)
            status = "✗ WRONG"
        
        image_name = test_dataset.imgs[idx][0].split('\\')[-1]
        
        print(f"{image_name:<40} {true_class:<35} {predicted_class:<35} {confidence_val:>6.2f}%      {status}")

print("="*80)

test_acc = 100 * correct / total

print(f"\n{'='*80}")
print(f"OVERALL TEST RESULTS")
print(f"{'='*80}")
print(f"Total Images: {total}")
print(f"Correct Predictions: {correct}")
print(f"Incorrect Predictions: {total - correct}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"\nAverage Confidence: {np.mean(confidence_scores):.2f}%")
print(f"Average Confidence (Correct): {np.mean(correct_confidences):.2f}%")
print(f"Average Confidence (Incorrect): {np.mean(incorrect_confidences):.2f}%")
print(f"{'='*80}\n")

print(f"{'='*80}")
print(f"PER-CLASS ACCURACY")
print(f"{'='*80}")
print(f"{'Class Name':<40} {'Accuracy':<12} {'Avg Confidence':<15} {'Samples'}")
print("-"*80)

for class_name in sorted(per_class_stats.keys()):
    stats = per_class_stats[class_name]
    class_acc = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0 #type: ignore
    avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
    print(f"{class_name:<40} {class_acc:>6.2f}%      {avg_conf:>6.2f}%          {stats['total']}")

print("="*80)