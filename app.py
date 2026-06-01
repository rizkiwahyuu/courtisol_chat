"""
Cortisoul Chat Service
Chatbot yang terhubung ke:
- Riwayat jurnal
- Hasil prediksi
- Refleksi AI
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================================================
# CONFIG
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_gemini_client = None

# =====================================================
# DATABASE (contoh sementara)
# Ganti dengan MySQL/PostgreSQL Anda
# =====================================================

FAKE_DB = {
    1: [
        {
            "text": "Aku merasa sangat lelah dan sulit tidur.",
            "label": "stress",
            "stress_score": 8.2,
            "kategori": "Berat",
            "refleksi_ai": (
                "Kamu sedang menghadapi tekanan yang cukup besar. "
                "Walaupun terasa berat, kamu masih berusaha menjalani hari."
            ),
        },
        {
            "text": "Hari ini sedikit lebih baik daripada kemarin.",
            "label": "stress",
            "stress_score": 6.8,
            "kategori": "Sedang",
            "refleksi_ai": (
                "Ada tanda bahwa kamu mulai menemukan ruang bernapas "
                "di tengah kesibukanmu."
            ),
        },
    ]
}

# =====================================================
# GEMINI
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gemini_client

    genai.configure(api_key="GEMINI_API_KEY")

    _gemini_client = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config={
            "temperature": 0.8,
            "max_output_tokens": 200,
        }
    )

    logger.info("Gemini siap")
    yield


app = FastAPI(
    title="Cortisoul Chat Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# SCHEMA
# =====================================================

class ChatRequest(BaseModel):
    user_id: int
    message: str


class ChatResponse(BaseModel):
    reply: str
    model: str


# =====================================================
# DATABASE FUNCTION
# =====================================================

def get_user_history(user_id: int, limit: int = 5):

    history = FAKE_DB.get(user_id, [])

    return history[-limit:]


# =====================================================
# PROMPT BUILDER
# =====================================================

def build_prompt(
    user_message: str,
    history: List[dict]
):

    context = ""

    for i, item in enumerate(history, start=1):

        context += f"""

=== RIWAYAT {i} ===

JURNAL:
{item['text']}

HASIL ANALISIS:
Label: {item['label']}
Stress Score: {item['stress_score']}
Kategori: {item['kategori']}

REFLEKSI AI:
{item['refleksi_ai']}

"""

    prompt = f"""
Kamu adalah Cortisoul.

Kamu adalah teman refleksi yang hangat,
empatik, dan suportif.

Berikut riwayat jurnal pengguna:

{context}

Tugasmu:

- Pahami pola emosi pengguna.
- Gunakan refleksi sebelumnya sebagai konteks.
- Jangan mengulang refleksi lama mentah-mentah.
- Berikan jawaban yang terasa nyambung.
- Fokus membantu pengguna memahami dirinya.
- Gunakan bahasa Indonesia.
- Maksimal 200 kata.

Pesan terbaru pengguna:

"{user_message}"

Berikan respons yang personal dan relevan.
"""

    return prompt


# =====================================================
# GEMINI CHAT
# =====================================================

async def ask_gemini(prompt: str):

    response = await asyncio.to_thread(
        _gemini_client.generate_content,
        prompt
    )

    return response.text.strip()


# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": GEMINI_MODEL
    }


@app.post("/chat")
async def chat(body: ChatRequest):

    history = get_chat_history(body.user_id)

    gemini_history = []

    for msg in history:
        gemini_history.append({
            "role": msg["role"],
            "parts": [msg["message"]]
        })

    chat_session = _gemini_client.start_chat(
        history=gemini_history
    )

    response = await asyncio.to_thread(
        chat_session.send_message,
        body.message
    )

    save_message(
        body.user_id,
        "user",
        body.message
    )

    save_message(
        body.user_id,
        "assistant",
        response.text
    )

    return {
        "reply": response.text
    }