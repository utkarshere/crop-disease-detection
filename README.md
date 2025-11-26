# 🌾 Crop Disease Detection API

**Deep Learning + RAG + LLM-based Agronomist Recommendations**

This project provides an end-to-end pipeline for detecting crop/leaf diseases from an image and delivering actionable treatment guidance.

It combines:

* 📷 **EfficientNet-B0** image classifier
* 🔍 **RAG (Retrieval-Augmented Generation)** using Sentence Transformers
* 🤖 **LLM (GPT-4o-mini)** to rewrite treatment advice like an expert agronomist
* ⚡ **FastAPI backend** for production-ready serving
* 🌐 **CORS-enabled API** for frontend integration

---

## 🚀 Features

### 🌱 **1. Disease Prediction**

Upload a crop/leaf image → Model returns:

* predicted disease
* confidence score (0–100%)
* normalized class name

### 📚 **2. RAG-based Treatment Retrieval**

The predicted disease is embedded & matched with your knowledge base:

* `documents.json` — original treatment text
* `vector_store.json` — precomputed embeddings

Closest disease document is retrieved using cosine similarity.

### 🧠 **3. Expert Agronomist Treatment Rewrite**

Once the correct treatment text is retrieved, an LLM generates:

* simple explanation of disease
* early warning signs
* immediate farmer actions
* recommended fungicides (non-jargon)
* future prevention advice

### 🌍 **4. Multilingual Output (Optional)**

Define any output language — Hindi, Marathi, Spanish, etc.

(You can extend API to support `lang="hi"`.)

---

## 📑 Project Structure

`<pre class="overflow-visible!" data-start="1742" data-end="2286"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary">``<div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2">``<div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div>``</div></div>``<div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!">``<span><span>`Crop_Detection/
│── src/
│   ├── app.py                  `<span>`# FastAPI application (endpoints)`<span>`
│   ├── api_model_loader.py     `<span>`# Model loading + prediction + RAG `<span>`
│   └── frontend/               `<span>`# Optional UI `<span>`
│
│── knowledge_base/
│   ├── documents.json          `<span>`# Dummy Treatment database `<span>`
│   └── vector_store.json       `<span>`# Generated embeddings `<span>`
│
│── models/
│   └── best_model.pt           `<span>`# EfficientNet-B0 trained model `<span>`
│
│── dataset_split/
│   └── `<span>`test `<span>`/                   `<span>`# Small test dataset for inference test `<span>`
│
│── requirements.txt
│── README.md
`</code></div>``</div></pre>`

---

## ⚙️ Setup Instructions

### 1. Clone the repository

<pre class="overflow-visible!" data-start="2348" data-end="2448"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>git </span><span>clone</span><span> https://github.com/utkarshere/crop-disease-detection.git
</span><span>cd</span><span> crop-disease-detection
</span></span></code></div></div></pre>

### 2. Install dependencies

<pre class="overflow-visible!" data-start="2479" data-end="2518"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>pip</span><span> install -r requirements.txt
</span></span></code></div></div></pre>

### 3. Add your OpenAI key

Create `.env`:

<pre class="overflow-visible!" data-start="2564" data-end="2600"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>OPENAI_API_KEY</span><span>=your_key_here
</span></span></code></div></div></pre>

### 4. Run the API

<pre class="overflow-visible!" data-start="2622" data-end="2658"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>uvicorn </span><span>src</span><span>.app</span><span>:app --reload
</span></span></code></div></div></pre>

API runs at:

👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

Swagger UI:

👉 **[http://127.0.0.1:8000/docs]()**

---

## 🧪 API Endpoints

### **1. Health Check**

`GET /`

Returns:

<pre class="overflow-visible!" data-start="2824" data-end="2907"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"status"</span><span>:</span><span></span><span>"ok"</span><span>,</span><span>
  </span><span>"message"</span><span>:</span><span></span><span>"Crop Disease Detection API running"</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

### **2. Predict + Recommend (LLM Expert Advice)**

👉 **Single combined endpoint**

`POST /predict_and_recommend_expert`

Form-data:

<pre class="overflow-visible!" data-start="3049" data-end="3070"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>file: <image>
</span></span></code></div></div></pre>

Response Example:

<pre class="overflow-visible!" data-start="3091" data-end="3328"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"predicted_disease"</span><span>:</span><span></span><span>"Apple rust leaf"</span><span>,</span><span>
  </span><span>"confidence"</span><span>:</span><span></span><span>"96.21%"</span><span>,</span><span>
  </span><span>"matched_disease"</span><span>:</span><span></span><span>"Apple_rust_leaf"</span><span>,</span><span>
  </span><span>"similarity_score"</span><span>:</span><span></span><span>0.91</span><span>,</span><span>
  </span><span>"raw_treatment"</span><span>:</span><span></span><span>"..."</span><span>,</span><span>
  </span><span>"expert_recommendation"</span><span>:</span><span></span><span>"Farmer-friendly explanation..."</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

## 🧠 ML Pipeline Overview

### **Image Classification**

* EfficientNet-B0
* Trained for 20 epochs on Kaggle P100 GPU
* Data augmentation via `transforms`
* Saves best model as `best_model.pt`

### **RAG Retrieval**

* Embeddings generated using `all-MiniLM-L6-v2`
* Stores embeddings per disease in JSON
* Cosine similarity → best matching document

### **LLM Expert Rewrite**

* Model: `gpt-4o-mini`
* Role: “Expert agronomist”
* Converts raw treatment into actionable farmer guidance

Multilingual support can be added by passing `lang=<code>`.

---

## 🧪 Testing the API With Sample Data

A small `dataset_split/test/` folder is provided so users can:

* test predictions
* validate confidence scores
* check treatment recommendations

---

## 📦 Deployment

You can deploy the API using:

* **Render**
* **Railway**
* **Docker**
* **AWS EC2**
* **Azure App Service**
* **Google Cloud Run**

Add this for Docker:

<pre class="overflow-visible!" data-start="4264" data-end="4347"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>docker build -t crop-disease-api .
docker run -p </span><span>8000</span><span>:</span><span>8000</span><span> crop-disease-api
</span></span></code></div></div></pre>

---

## ⭐ Support

If you found this useful, please ⭐ star the repository!
