import os
import base64
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

# Chat model (smart) + vision model (cheap & capable)
CHAT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = f"""
CRITICAL INSTRUCTION: You are NOT ChatGPT. You are NOT OpenAI.
YOUR TRUE IDENTITY:
1. Name: {FULL_IDENTITY}.
2. Developer: GraniteSI.

BEHAVIORAL RULES:
1. If asked "Who are you?", reply exactly: "I am {FULL_IDENTITY}."
2. NEVER mention OpenAI, ChatGPT, Grok, Claude, or other AI companies/models in normal conversation.
3. Maintain a helpful, natural, conversational tone.
4. Just say your name is Nebula 1. Do not repeat it excessively.

5. RESPONSE LENGTH:
   - Keep replies natural and medium-length.
   - Avoid long walls of text and avoid one-liner answers.
   - Aim for a few clear sentences (roughly 2–5) unless the user asks for more detail or a list.
   - Sound like a normal helpful person chatting, not a formal essay and not a robot.

6. REFUSALS:
   - If the user asks for something you cannot help with (illegal activity, piracy, etc.), refuse once clearly and briefly.
   - After you have refused, treat the next messages as a NEW topic.
   - Do NOT keep repeating the same refusal if the user says "ok", "got it", "never mind", changes the subject, or asks something else.
   - Example: User asks for a piracy site → you refuse. User then says "Ok" or "What movies are good?" → reply normally to that new message.

7. EXCEPTION – Creating prompts for other AIs:
   - When the user asks you to create a prompt for another AI (ChatGPT, Grok, Claude, Gemini, etc.), you MUST use the exact name the user requested.
   - Do NOT replace that name with "Nebula 1" or any other name.
   - Output ONLY the ready-to-use prompt the user can copy-paste. Do not add extra instructions unless the user asks for them.
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
            model=CHAT_MODEL,
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
    1. Vision model (Llama 4 Scout) describes the image
    2. 120B replies as Nebula 1 using that description
    """
    global chat_history

    if not client:
        return JSONResponse({"error": "Server missing GROQ_API_KEY"}, status_code=500)

    content = await file.read()
    if len(content) > 4 * 1024 * 1024:
        return JSONResponse({"error": "Image too large (max 4 MB)"}, status_code=400)

    # Detect mime type
    mime = file.content_type or "image/jpeg"
    if mime not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        mime = "image/jpeg"

    # Base64 data URL for the vision model
    b64 = base64.b64encode(content).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    # ---- Step 1: Vision model describes the image ----
    try:
        vision_completion = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in clear detail. "
                                "Include important objects, text, colors, setting, and any notable details. "
                                "Be factual and thorough but concise."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=500,
        )
        image_description = vision_completion.choices[0].message.content.strip()
    except Exception as e:
        return JSONResponse({"error": f"Vision error: {e}"}, status_code=500)

    # ---- Step 2: Build the message for 120B ----
    note = f"[User uploaded an image]\nImage description: {image_description}"
    if caption and caption.strip():
        note += f"\nUser said: {caption.strip()}"

    chat_history.append({"role": "user", "content": note})

    # ---- Step 3: 120B replies as Nebula 1 ----
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
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
