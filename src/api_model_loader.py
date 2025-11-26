import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pt"
DOCUMENTS_PATH = ROOT / "knowledge_base" / "documents.json"
VECTOR_STORE_PATH = ROOT / "knowledge_base" / "vector_store.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_name(name: str) -> str:
    """Normalize names to lowercase underscore form for robust matching."""
    if not name:
        return ""
    s = name.strip().lower()
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def load_cnn_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["classes"]
    num_classes = len(class_names)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes) #type: ignore

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names

MODEL, CLASS_NAMES = load_cnn_model(MODEL_PATH)


IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
    DOCUMENTS: Dict[str, Any] = json.load(f)

with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
    VECTOR_STORE_RAW: Dict[str, Any] = json.load(f)


DOC_KEY_MAP: Dict[str, str] = {}
for k in DOCUMENTS.keys():
    DOC_KEY_MAP[normalize_name(k)] = k


EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

AVG_EMBEDDINGS: Dict[str, np.ndarray] = {}
for disease_key, embeds in VECTOR_STORE_RAW.items():
    if not embeds:
        continue
    arr = np.array(embeds, dtype=np.float32)
    AVG_EMBEDDINGS[disease_key] = arr.mean(axis=0)


def predict_disease_from_tensor(image_tensor: torch.Tensor) -> Tuple[str, float]:
    """Return (label_str, confidence_percent)"""
    MODEL.eval()
    img = image_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(img)
        probs = F.softmax(outputs, dim=1)
        conf_val, pred_idx = probs.max(dim=1)
        label = CLASS_NAMES[pred_idx.item()]
        conf = float(conf_val.item() * 100.0)
    return label, float(f"{conf:.2f}")

def predict_from_pil(pil_image) -> Tuple[str, float]:
    tensor = IMG_TRANSFORM(pil_image)
    return predict_disease_from_tensor(tensor)


def embed_text(text: str) -> np.ndarray:
    emb = EMBEDDING_MODEL.encode(text)
    return np.array(emb, dtype=np.float32)

def find_best_matching_disease_by_similarity(predicted_text: str) -> Tuple[Optional[str], float]:
    """Return (best_disease_key, similarity_score). Uses precomputed AVG_EMBEDDINGS."""
    if not AVG_EMBEDDINGS:
        return None, 0.0

    query_emb = embed_text(predicted_text)
    best_score = -1.0
    best_key = None

    for disease_key, avg_emb in AVG_EMBEDDINGS.items():
     
        score = util.cos_sim(torch.from_numpy(query_emb), torch.from_numpy(avg_emb)).item()
        if score > best_score:
            best_score = score
            best_key = disease_key

    return best_key, float(best_score)

def get_treatment_text_for_key(disease_key: str) -> str:
    if disease_key not in DOCUMENTS:
        return ""
    docs = DOCUMENTS[disease_key]
    pieces = []
    for d in docs:
        title = d.get("title", "").strip()
        content = d.get("content", "").strip()
        if title:
            pieces.append(f"{title}:\n{content}")
        else:
            pieces.append(content)
    return "\n\n".join(pieces)

def get_all_available_categories() -> List[str]:
    """Return sorted list of available document keys (original names)."""
    return sorted(list(DOCUMENTS.keys()))


SIMILARITY_THRESHOLD = 0.80  

def get_treatment_for_prediction(predicted_label: str) -> Tuple[Optional[str], float, str]:
    """
    Hybrid matching:
      1) Normalize predicted_label; if exact normalized key exists in DOC_KEY_MAP -> direct match.
      2) Else run RAG similarity. Accept only if similarity >= SIMILARITY_THRESHOLD.
      3) Otherwise return (None, best_score, "")
    Returns: (matched_document_key_or_None, similarity_score, treatment_text)
    """
    if not predicted_label:
        return None, 0.0, ""

    pred_norm = normalize_name(predicted_label)

    
    if pred_norm in DOC_KEY_MAP:
        matched_key = DOC_KEY_MAP[pred_norm]
        treatment = get_treatment_text_for_key(matched_key)
        return matched_key, 1.0, treatment

    
    best_key, best_score = find_best_matching_disease_by_similarity(predicted_label)
    if best_key and best_score >= SIMILARITY_THRESHOLD:
        treatment = get_treatment_text_for_key(best_key)
        return best_key, best_score, treatment

   
    return None, best_score if best_score is not None else 0.0, ""
