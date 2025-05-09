from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional
from model import Model
import os
from dotenv import load_dotenv

load_dotenv()  
app = FastAPI()

# Static files (images, CSS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates (renamed from 'templates' to 'pages')
templates = Jinja2Templates(directory="pages")

# Initialize your AI model
MODEL_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.0-flash-001"
API_KEY = os.getenv("API_KEY")  # Replace with your actual key
print(API_KEY)

client = Model(MODEL_BASE_URL, MODEL, API_KEY)

# Persona definitions
personas = {
    "san":   {"menu_name": "Ju’hoansi (San)", "name": "!Kunta Xa", "avatar": "san.png", "prompt": "san"},
    "zapo":  {"menu_name": "Zapotec", "name": "Bianu Xquidxe", "avatar": "zapo.png", "prompt": "zapo"},
    "yano":  {"menu_name": "Yanomami", "name": "Tëpɨ Yanoama", "avatar": "yano.png", "prompt": "yano"},
    "ainu":  {"menu_name": "Ainu", "name": "Rikusa Samay", "avatar": "ainu.png", "prompt": "ainu"},
    "nawe":  {"menu_name": "Enawene Nawe", "name": "Kalari Wakaliti", "avatar": "nawe.png", "prompt": "nawe"},
}

def get_prompt(persona_key: str) -> str:
    path = Path(f"prompts/{persona_key}.txt")
    return path.read_text(encoding="utf-8")

# --- Route: Home Page ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "personas": personas
    })

@app.get("/chat", response_class=HTMLResponse)
async def load_chat(request: Request, persona_key: str):
    persona = personas.get(persona_key)
    if not persona:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "personas": personas,
            "error": "Invalid persona selected."
        })

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "persona": persona,
        "persona_key": persona_key,
        "user_input": None,
        "real_language": None,
        "translated_language": None
    })

@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    persona_key: str = Form(...),
    message: Optional[str] = Form(default=None)
):
    persona = personas.get(persona_key)
    if not persona or not message or not message.strip():
        return templates.TemplateResponse("chat.html", {
            "request": request,
            "persona": persona,
            "persona_key": persona_key,
            "user_input": "",
            "real_language": None,
            "translated_language": None
        })

    reply = client.fetch_response(message, persona_key)
    parts = reply.split("||")
    real_text = parts[0].strip()
    english = parts[1].strip()

    return HTMLResponse(content=f"""
        <div class="message">
            <div class="bubble-meta">{persona['name']}:</div>
            <span id="english-response">{english}</span>
            <div id="real-language" class="translated" style="display: none;">{real_text}</div>
        </div>
    """)