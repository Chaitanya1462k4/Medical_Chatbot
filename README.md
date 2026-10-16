# 🩺 Medical Chatbot — RAG-powered Q&A Assistant

A complete end-to-end **Medical Chatbot** built using **FastAPI**, **LangChain**, **Pinecone**, and **OpenRouter**. Ask medical questions and get concise, context-aware answers retrieved from medical literature.

---

## 📸 Demo

> Ask about symptoms, conditions, medications, and treatments — all backed by medical literature via RAG (Retrieval-Augmented Generation).

---

## ⚙️ How to Run?

### STEPS:

### Clone the repository

```bash
git clone https://github.com/your-username/Medical_Chatbot.git
```

---

### STEP 01 — Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.11 -y
conda activate medibot
```

---

### STEP 02 — Install the requirements

```bash
pip install -r requirements.txt
```

---

### STEP 03 — Set up environment variables

Create a `.env` file in the root directory and add your **Pinecone** & **OpenRouter** credentials:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENROUTER_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

> 🔑 Get your keys from:
> - **Pinecone**: https://www.pinecone.io/
> - **OpenRouter**: https://openrouter.ai/keys

---

### STEP 04 — Store embeddings to Pinecone

Run the following command **once** to load the medical PDF, chunk it, embed it, and push it to Pinecone:

```bash
python Store_index.py
```

---

### STEP 05 — Run the application

```bash
uvicorn app:app --reload
```

Now open your browser and navigate to:

```
http://localhost:8000
```

Interactive API docs are available at:

```
http://localhost:8000/docs
```

---

## 🧰 Tech Stack Used

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Framework | FastAPI |
| LLM Orchestration | LangChain |
| Vector Database | Pinecone |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM | OpenRouter (NVIDIA Nemotron) |
| Frontend | HTML · CSS · Vanilla JS |
| Server | Uvicorn |

---

## 🏗️ Project Structure

```
Medical_Chatbot/
├── data/
│   └── Medical_book.pdf        ← Knowledge base
├── research/
│   └── trials.ipynb            ← Experimentation notebook
├── src/
│   ├── __init__.py
│   ├── helper.py               ← PDF loading, chunking, embeddings
│   └── prompt.py               ← System prompt
├── static/
│   └── style.css               ← Chat UI styling
├── templates/
│   └── index.html              ← Chat UI template
├── app.py                      ← FastAPI application
├── Store_index.py              ← One-time PDF ingestion script
├── setup.py
├── requirements.txt
├── .env                        ← API keys (not committed)
└── .gitignore
```

---

## 🔄 Architecture

```
Medical_book.pdf
    └─► DirectoryLoader
        └─► RecursiveCharacterTextSplitter  (chunk=500, overlap=20)
            └─► HuggingFace Embeddings  (all-MiniLM-L6-v2, dim=384)
                └─► Pinecone Vector Store  (cosine similarity)
                          │
                          ▼  similarity search (k=3)
User Query ──► FastAPI /chat ──► LangChain LCEL Chain ──► OpenRouter LLM ──► Answer
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Chat UI |
| `POST` | `/chat`  | Send a question, receive an answer |
| `GET`  | `/health`| RAG chain health check |
| `GET`  | `/docs`  | Swagger interactive API docs |

### `POST /chat` — Example

**Request**
```json
{
  "message": "What are the symptoms of diabetes?"
}
```

**Response**
```json
{
  "answer": "Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision...",
  "sources": ["data/Medical_book.pdf"]
}
```

---

## 🚀 AWS CI/CD Deployment with GitHub Actions

### 1. Login to AWS Console

### 2. Create IAM User for Deployment

Grant the following specific access:

- **EC2 access** — virtual machine to run the app
- **ECR** — Elastic Container Registry to store your Docker image

### 3. About the Deployment Flow

```
1. Build Docker image of the source code
2. Push Docker image to ECR
3. Launch an EC2 instance (Ubuntu)
4. Pull the image from ECR into EC2
5. Launch the Docker container in EC2
```

### 4. IAM Policies Required

```
AmazonEC2ContainerRegistryFullAccess
AmazonEC2FullAccess
```

### 5. Create ECR Repository

Store/save your Docker image. Save the URI, e.g.:

```
315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot
```

### 6. Create EC2 Machine (Ubuntu)

Open EC2 and install Docker:

```bash
# Optional updates
sudo apt-get update -y
sudo apt-get upgrade -y

# Required — install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### 7. Configure EC2 as Self-Hosted GitHub Actions Runner

```
GitHub Repo → Settings → Actions → Runners → New self-hosted runner
→ Choose OS → Run the commands one by one
```

### 8. Set Up GitHub Secrets

Add the following secrets in your GitHub repository:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
PINECONE_API_KEY
OPENROUTER_API_KEY
```

---

## ⚠️ Disclaimer

> This chatbot is for **informational purposes only** and does **not** constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

---

## 🙏 Acknowledgements

Inspired by [entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS](https://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS).  
Adapted to use **FastAPI** and **OpenRouter** instead of Flask and OpenAI.
