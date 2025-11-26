import torch
import torch.nn.functional as F
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import json

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pt"
VECTOR_STORE_PATH = ROOT / "knowledge_base" / "vector_store.json"
DOCUMENTS_PATH = ROOT / "knowledge_base" / "documents.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PlantDiseaseRAG:
    def __init__(self):
        self.device = DEVICE
        self.load_classification_model()
        self.load_embedding_model()
        self.load_knowledge_base()
    
    def load_classification_model(self):
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        self.class_names = checkpoint['classes']
        num_classes = len(self.class_names)
        
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,  #type: ignore
            num_classes
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_embedding_model(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def load_knowledge_base(self):
        if DOCUMENTS_PATH.exists():
            with open(DOCUMENTS_PATH, 'r') as f:
                self.documents = json.load(f)
        else:
            print(f"Warning: {DOCUMENTS_PATH} not found. Using empty knowledge base.")
            self.documents = {}
        
        if VECTOR_STORE_PATH.exists():
            with open(VECTOR_STORE_PATH, 'r') as f:
                data = json.load(f)
                self.document_embeddings = {k: np.array(v) for k, v in data.items()}
        else:
            print("Creating vector store from documents...")
            self.create_vector_store()
    
    def create_vector_store(self):
        self.document_embeddings = {}
        
        for disease, docs in self.documents.items():
            disease_embeddings = []
            for doc in docs:
                text = f"{doc['title']} {doc['content']}"
                embedding = self.embedding_model.encode(text)
                disease_embeddings.append(embedding.tolist())
            self.document_embeddings[disease] = disease_embeddings
        
        VECTOR_STORE_PATH.parent.mkdir(exist_ok=True, parents=True)
        with open(VECTOR_STORE_PATH, 'w') as f:
            json.dump({k: v for k, v in self.document_embeddings.items()}, f)
        
        print(f"Vector store saved to {VECTOR_STORE_PATH}")
    
    def predict_disease(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device) #type: ignore
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        top5_prob, top5_idx = probabilities.topk(5)
        top5_predictions = [
            (self.class_names[idx.item()], prob.item() * 100)
            for idx, prob in zip(top5_idx[0], top5_prob[0])
        ]
        
        return predicted_class, confidence_score, top5_predictions
    
    def is_healthy_class(self, disease_name):
        healthy_keywords = ['healthy', 'Healthy', 'HEALTHY']
        return any(keyword in disease_name for keyword in healthy_keywords)
    
    def get_healthy_recommendation(self, plant_name):
        plant = plant_name.split('___')[0] if '___' in plant_name else plant_name.split('_')[0]
        
        recommendations = {
            'general': f"""
Your {plant} plant appears healthy! To maintain plant health:

1. PREVENTIVE MONITORING
   - Inspect plants weekly for early signs of disease or pests
   - Check both upper and lower leaf surfaces
   - Monitor new growth and flowering areas
   - Keep records of observations

2. CULTURAL PRACTICES
   - Maintain proper spacing for air circulation
   - Water at soil level, avoid wetting foliage
   - Water in early morning so leaves dry quickly
   - Apply mulch to prevent soil splash and suppress weeds

3. NUTRITION MANAGEMENT
   - Provide balanced fertilization based on soil tests
   - Avoid excessive nitrogen which promotes disease
   - Ensure adequate potassium and micronutrients
   - Monitor for nutrient deficiency symptoms

4. SANITATION
   - Remove plant debris and fallen leaves regularly
   - Clean tools between uses
   - Remove any diseased plant material immediately
   - Practice crop rotation where applicable

5. PEST MANAGEMENT
   - Scout regularly for insect pests
   - Encourage beneficial insects
   - Use physical barriers like row covers when appropriate
   - Apply organic or chemical controls only when thresholds are reached

Continue these practices to keep your {plant} plants disease-free and productive!
"""
        }
        
        return recommendations['general']
    
    def retrieve_treatment(self, disease_name, query="treatment and prevention", top_k=3):
        if disease_name not in self.documents:
            return []
        
        query_embedding = self.embedding_model.encode(query)
        disease_embeddings = self.document_embeddings[disease_name]
        
        similarities = []
        for idx, doc_embedding in enumerate(disease_embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((idx, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = []
        for idx, score in similarities[:top_k]:
            doc = self.documents[disease_name][idx]
            top_docs.append({
                'title': doc['title'],
                'content': doc['content'],
                'relevance_score': score * 100
            })
        
        return top_docs
    
    def diagnose_and_treat(self, image_path):
        predicted_disease, confidence, top5 = self.predict_disease(image_path)
        
        print("="*80)
        print("DISEASE DIAGNOSIS")
        print("="*80)
        print(f"Predicted Disease: {predicted_disease}")
        print(f"Confidence: {confidence:.2f}%\n")
        
        print("Top 5 Predictions:")
        for rank, (disease, prob) in enumerate(top5, 1):
            print(f"  {rank}. {disease}: {prob:.2f}%")
        
        print("\n" + "="*80)
        print("TREATMENT RECOMMENDATIONS")
        print("="*80)
        
        if self.is_healthy_class(predicted_disease):
            print("\n✓ HEALTHY PLANT DETECTED - No Treatment Required\n")
            print(self.get_healthy_recommendation(predicted_disease))
            treatments = []
        else:
            treatments = self.retrieve_treatment(predicted_disease)
            
            if not treatments:
                print(f"No treatment information available for {predicted_disease}")
            else:
                for idx, doc in enumerate(treatments, 1):
                    print(f"\n{idx}. {doc['title']} (Relevance: {doc['relevance_score']:.2f}%)")
                    print("-" * 80)
                    print(doc['content'])
        
        print("="*80)
        
        return {
            'predicted_disease': predicted_disease,
            'confidence': confidence,
            'top5_predictions': top5,
            'is_healthy': self.is_healthy_class(predicted_disease),
            'treatments': treatments
        }

if __name__ == "__main__":
    rag_system = PlantDiseaseRAG()
    
    test_image = ROOT / "dataset_split" / "test" / "Apple___Apple_scab" / "image_001.jpg"
    
    if test_image.exists():
        result = rag_system.diagnose_and_treat(test_image)
    else:
        print(f"Test image not found: {test_image}")
        print("Please provide a valid image path")