import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

from src.api_model_loader import (
    IMG_TRANSFORM,
    predict_from_pil,
    get_treatment_for_prediction,
    get_all_available_categories
)

app = FastAPI(title="Crop Disease Detection API (Hybrid match + RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    disease_name: str
    language: str = "english"  

@app.get("/")
def root():
    return {"status": "ok", "message": "Crop Disease Detection API running"}

@app.get("/categories")
def list_categories():
    """
    Returns the list of available diseases/crops present in knowledge_base/documents.json.
    Useful for frontend dropdown suggestions.
    """
    categories = get_all_available_categories()
    return {
        "count": len(categories),
        "categories": categories
    }


@app.post("/predict_and_recommend_expert")
async def predict_and_recommend_expert(file: UploadFile = File(...)):
   
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

  
    predicted_label, confidence = predict_from_pil(pil_image)
    confidence_str = f"{confidence:.2f}%"

    
    matched_key, similarity, treatment_raw = get_treatment_for_prediction(predicted_label)
    similarity_val = float(f"{similarity:.4f}")

    if not matched_key or not treatment_raw:
        categories = get_all_available_categories()
        friendly_list = "\n".join(f"- {c}" for c in categories)
        message = (
            "Could not confidently identify this leaf/crop. "
            "Please upload a clear leaf image of one of the following categories:\n\n"
            f"{friendly_list}"
        )
        return {
            "predicted_disease": predicted_label,
            "confidence": confidence_str,
            "matched_disease": None,
            "similarity_score": similarity_val,
            "raw_treatment": "",
            "expert_recommendation": "",
            "note": message
        }

    
    if matched_key == "Healthy":
        expert_advice = treatment_raw
    else:
        system_msg = "You are a concise, practical agronomist. Provide farmer-friendly, actionable recommendations."
        user_prompt = f"""
    Rewrite the treatment information below for a farmer. Keep it simple, practical and actionable.
    Include:
    1) What the disease is (short)
    2) Early warning signs
    3) Immediate actions
    4) Recommended treatments (chemical/organic simple guidance)
    5) Preventive measures for future seasons
    
    Original treatment text:
    {treatment_raw}
    """
    
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(
                f"{system_msg}\n\n{user_prompt}"
            )
            expert_advice = response.text
        except Exception as e:
            # Fallback if LLM fails
            expert_advice = ""
            llm_error_note = f"LLM generation failed: {e}"
            return {
                "predicted_disease": predicted_label,
                "confidence": confidence_str,
                "matched_disease": matched_key,
                "similarity_score": similarity_val,
                "raw_treatment": treatment_raw,
                "expert_recommendation": expert_advice,
                "note": "Expert rewrite failed. See raw_treatment. " + llm_error_note,
                "is_healthy": False
            }

    return {
        "predicted_disease": predicted_label,
        "confidence": confidence_str,
        "matched_disease": matched_key,
        "similarity_score": similarity_val,
        "raw_treatment": treatment_raw,
        "expert_recommendation": expert_advice,
        "is_healthy": (matched_key == "Healthy")
    }
