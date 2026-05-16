# 🏥 MediAssist — AI-Powered Medical Chatbot

An intelligent medical chatbot built with **LangChain**, **Pinecone**, **Ollama (Llama 3.2)**, and **Flask**. It provides symptom-based medical guidance, prescription image analysis (OCR), and persistent chat history — all wrapped in a sleek, modern UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey?logo=flask)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Medical Q&A** | RAG-powered answers using medical textbook data + real doctor-patient conversations |
| 📋 **Prescription OCR** | Upload a prescription image → get a full plain-language breakdown via NVIDIA Nemotron VL |
| 🔐 **User Authentication** | Register/Login with username & password, or sign in with **Google OAuth 2.0** |
| 👤 **User Profiles** | Age & gender stored to personalize medical advice |
| 💬 **Chat History** | All conversations (including prescriptions) persist and are viewable from the sidebar |
| 🗑️ **History Management** | Delete individual conversations or clear all history |
| 🎤 **Voice Input** | Speech-to-text via Web Speech API |
| 🔊 **Text-to-Speech** | Bot reads responses aloud (toggleable) |
| 🏷️ **Symptom Categories** | Quick-select symptom categories (Head, Heart, Stomach, Mental Health, etc.) |
| 📱 **Responsive UI** | Modern dark-themed interface with animations, markdown rendering, and action buttons |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Flask Web App (app.py)                  │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ Auth     │  │ Chat (RAG)   │  │ Prescription OCR      │ │
│  │ Routes   │  │ /get         │  │ /analyze-prescription  │ │
│  └────┬─────┘  └──────┬───────┘  └──────────┬────────────┘ │
│       │               │                     │               │
│  ┌────▼─────┐  ┌──────▼───────┐  ┌──────────▼────────────┐ │
│  │ UserDB   │  │ Ollama       │  │ NVIDIA NIM API        │ │
│  │(Pinecone)│  │ (Llama 3.2)  │  │ (Nemotron VL 8B)      │ │
│  └──────────┘  └──────────────┘  └───────────────────────┘ │
│                       │                                     │
│              ┌────────▼─────────┐                           │
│              │ Pinecone Index   │                           │
│              │ "medical-chatbot"│                           │
│              │                  │                           │
│              │ ns=""  (medical) │                           │
│              │ ns="users"      │                           │
│              │ ns="chat_hist"  │                           │
│              └──────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 How to Run

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running locally ([install guide](https://ollama.com/download))
- A **Pinecone** account (free tier works)
- An **NVIDIA NIM API key** for prescription OCR ([get one free](https://build.nvidia.com))
- *(Optional)* Google OAuth credentials for social login

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Sunnykumar1554/Medical-Chatbot.git
cd Medical-Chatbot
```

### Step 2 — Create a Conda Environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Pull the Ollama Model

```bash
ollama pull llama3.2:1b
```

### Step 5 — Configure Environment Variables

Create a `.env` file in the root directory:

```ini
PINECONE_API_KEY = "your-pinecone-api-key"
LLAMA_MODEL = "llama3.2:1b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LLAMA_TEMPERATURE = "0.1"
FLASK_SECRET_KEY = "any-random-string-here"

# NVIDIA API Key (for prescription OCR)
NVIDIA_API_KEY = "nvapi-your-nvidia-api-key"

# Google OAuth 2.0 (optional — for Google sign-in)
GOOGLE_CLIENT_ID = "your-google-client-id"
GOOGLE_CLIENT_SECRET = "your-google-client-secret"
```

### Step 6 — Index Medical Data into Pinecone

```bash
python store_index.py
```

### Step 7 — Run the App

```bash
python app.py
```

Open your browser and go to **http://localhost:5000**

---

## 📁 Project Structure

```
Medical-Chatbot/
├── app.py                  # Main Flask application
├── user_db.py              # User auth & chat history (Pinecone)
├── store_index.py          # Index medical PDFs into Pinecone
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
├── Dockerfile              # Docker deployment config
├── .env                    # Environment variables (not committed)
├── src/
│   ├── __init__.py
│   ├── helper.py           # Embeddings, PDF/CSV loaders, text splitter
│   ├── prompt.py           # System prompts (chat + prescription)
│   └── store_csv_index.py  # Index CSV Q&A data into Pinecone
├── data/                   # Medical PDFs and CSV data
├── templates/
│   ├── chat.html           # Main chat interface
│   ├── login.html          # Login page
│   └── register.html       # Registration page
├── static/
│   └── style.css           # All styles (dark theme, animations)
└── research/               # Experiment notebooks
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML, CSS, JavaScript, jQuery, Marked.js |
| **Backend** | Flask (Python) |
| **LLM (Chat)** | Ollama + Llama 3.2 (1B) — runs locally |
| **LLM (OCR)** | NVIDIA Nemotron VL 8B via NIM API |
| **Vector DB** | Pinecone (medical knowledge + user data + chat history) |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2, 384-dim) |
| **RAG Framework** | LangChain (retrieval chain + MMR search) |
| **Auth** | Flask sessions + Google OAuth 2.0 (Authlib) |
| **Voice** | Web Speech API (STT + TTS) |

---

## 🔑 API Keys Required

| Key | Where to Get | Used For |
|-----|-------------|----------|
| `PINECONE_API_KEY` | [pinecone.io](https://www.pinecone.io/) | Vector storage for medical data, users & history |
| `NVIDIA_API_KEY` | [build.nvidia.com](https://build.nvidia.com) | Prescription image analysis (OCR) |
| `GOOGLE_CLIENT_ID` | [Google Cloud Console](https://console.cloud.google.com/apis/credentials) | *(Optional)* Google OAuth sign-in |
| `GOOGLE_CLIENT_SECRET` | Same as above | *(Optional)* Google OAuth sign-in |

---

## 📝 License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Sunny Kumar**
- GitHub: [@Sunnykumar1554](https://github.com/Sunnykumar1554)
- Email: sunny155415@gmail.com
