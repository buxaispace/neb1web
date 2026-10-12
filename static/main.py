import os
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.getenv("GROQ_API_KEY", "")
if not API_KEY:
    print("WARNING: GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=API_KEY) if API_KEY else None

MODEL_NAME = "Nebula 1"
FULL_IDENTITY = "Nebula 1"

SYSTEM_PROMPT = f"""
CRITICAL INSTRUCTION: You are NOT ChatGPT. You are NOT OpenAI.
YOUR TRUE IDENTITY:
1. Name: {FULL_IDENTITY}.
2. Developer: GraniteSI.
BEHAVIORAL RULES:
1. If asked "Who are you?", reply exactly: "I am {FULL_IDENTITY}."
2. NEVER mention OpenAI, ChatGPT, Grok, Claude, or other AI companies/models in normal conversation.
3. Maintain a helpful, conversational tone.
4. Just say your name is Nebula 1. Do not repeat it excessively.
5. EXCEPTION – Creating prompts for other AIs:
   - When the user asks you to create a prompt for another AI (ChatGPT, Grok, Claude, Gemini, etc.), you MUST use the exact name the user requested.
   - Do NOT replace that name with "Nebula 1" or any other name.
   - Output ONLY the ready-to-use prompt the user can copy-paste. Do not add extra instructions, explanations, or suggestions unless the user explicitly asks for them.
   - Example: If the user says "create a prompt for ChatGPT saying I am hungry", your entire reply should be the prompt itself that starts with something like "You are ChatGPT..." or the exact wording needed for that model.
"""

# In-memory chat history (resets when the free server sleeps)
chat_history: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

app = FastAPI(title="Nebula 1")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def sanitize_response(text: str) -> str:
    forbidden = [
        "large language model trained by",
        "As an AI",
        "language model trained by",
        "I am a large language model",
        "I'm a large language model",
    ]
    cleaned = text
    for phrase in forbidden:
        if phrase in cleaned:
            cleaned = cleaned.replace(phrase, MODEL_NAME)
    return cleaned


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_name": MODEL_NAME})


@app.post("/chat")
async def chat(message: str = Form(...)):
    global chat_history

    if not client:
        return JSONResponse({"error": "Server missing GROQ_API_KEY"}, status_code=500)

    user_text = message.strip()
    if not user_text:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    chat_history.append({"role": "user", "content": user_text})

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=chat_history,
            temperature=0.5,
        )
        raw = completion.choices[0].message.content
        final = sanitize_response(raw)
        chat_history.append({"role": "assistant", "content": final})
        return JSONResponse({"reply": final})
    except Exception as e:
        if len(chat_history) > 1:
            chat_history.pop()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...), caption: Optional[str] = Form(None)):
    """
    Accept an image from phone/tablet camera or file picker.
    For now we acknowledge the upload and optionally pass a text caption
    to the chat model. (True vision requires a vision-capable model.)
    """
    global chat_history

    if not client:
        return JSONResponse({"error": "Server missing GROQ_API_KEY"}, status_code=500)

    content = await file.read()
    # Keep under ~4 MB for free tier friendliness
    if len(content) > 4 * 1024 * 1024:
        return JSONResponse({"error": "Image too large (max 4 MB)"}, status_code=400)

    # Build a text note about the image so the model can respond
    note = f"[User uploaded an image: {file.filename or 'photo'}]"
    if caption and caption.strip():
        note += f"\nUser caption: {caption.strip()}"

    chat_history.append({"role": "user", "content": note})

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=chat_history,
            temperature=0.5,
        )
        raw = completion.choices[0].message.content
        final = sanitize_response(raw)
        chat_history.append({"role": "assistant", "content": final})
        return JSONResponse({
            "reply": final,
            "filename": file.filename,
            "size": len(content),
        })
    except Exception as e:
        if len(chat_history) > 1:
            chat_history.pop()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/reset")
async def reset():
    global chat_history
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    return JSONResponse({"status": "ok"})