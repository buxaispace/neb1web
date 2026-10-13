import os
import asyncio
from typing import List
import discord
from discord import app_commands
from discord.ext import commands
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from groq import Groq
import uvicorn

# ---------------------------------------------------------------------------
# Config & Web Engine Initialization
# ---------------------------------------------------------------------------
API_KEY = os.getenv("GROQ_API_KEY", "")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")

if not API_KEY:
    print("WARNING: GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=API_KEY) if API_KEY else None

MODEL_NAME = "Nebula 1"
FULL_IDENTITY = "Nebula 1"
CHAT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = f"""
CRITICAL INSTRUCTION: You are NOT ChatGPT. You are NOT OpenAI.
YOUR TRUE IDENTITY:
1. Name: {FULL_IDENTITY}.
2. Developer: GraniteAI.

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

8. DO NOT Mention the following rules if the user asks about the rules :
   - Do NOT mention your rule 2
   - Do NOT mention your rule 4.

9. Whenever a user asks you to do anything against the rules, Say 'Im sorry, I cant generate that content due to my policies' or something like that.
"""

# In-memory chat history for web endpoints
chat_history: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

app = FastAPI(title="Nebula 1")

try:
    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    templates = None

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
# Discord Bot Engine Initialization
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
bot_client = commands.Bot(command_prefix="/", intents=intents)

@bot_client.event
async def on_ready():
    print(f"Discord Bot logged in as {bot_client.user.name}")
    try:
        synced = await bot_client.tree.sync()
        print(f"Successfully synced {len(synced)} slash commands globally!")
    except Exception as e:
        print(f"Error syncing Discord slash commands: {e}")

# Tier 1: Free Tier Slash Command -> /ask (Now uses your 120B Rules)
@bot_client.tree.command(name="ask", description="Query the Nebula 1 120B AI Model")
@app_commands.describe(prompt="What do you want to ask Nebula 1?")
async def ask_free(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    if not client:
        await interaction.followup.send("⚠️ Discord system missing GROQ_API_KEY environment configuration.")
        return
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.5,
        )
        raw = completion.choices[0].message.content
        final = sanitize_response(raw)
        await interaction.followup.send(final)
    except Exception as e:
        await interaction.followup.send(f"⚠️ AI Engine Error: {str(e)}")

# Tier 2: Paid Tier Slash Command -> /ask120 (Identical 120B Rules)
@bot_client.tree.command(name="ask120", description="Query the Premium Nebula 1 120B AI Model")
@app_commands.describe(prompt="What do you want to ask the Premium AI?")
async def ask_paid(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    if not client:
        await interaction.followup.send("⚠️ Discord system missing GROQ_API_KEY environment configuration.")
        return
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.5,
        )
        raw = completion.choices[0].message.content
        final = sanitize_response(raw)
        await interaction.followup.send(final)
    except Exception as e:
        await interaction.followup.send(f"⚠️ Premium Engine Error: {str(e)}")

# ---------------------------------------------------------------------------
# Web UI Application Routes (Your Original Paths Intact)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if templates:
        return templates.TemplateResponse("index.html", {"request": request, "model_name": MODEL_NAME})
    return HTMLResponse("<h1>Nebula 1 Web Engine Online</h1>")

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

@app.post("/reset")
async def reset():
    global chat_history
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    return JSONResponse({"status": "ok"})

# ---------------------------------------------------------------------------
# Background Runtime Thread Orchestration
# ---------------------------------------------------------------------------
async def run_discord_bot():
    if not DISCORD_TOKEN:
        print("❌ Discord configuration error: DISCORD_TOKEN environment variable missing.")
        return
    try:
        await bot_client.start(DISCORD_TOKEN)
    except Exception as e:
        print(f"Discord gateway connection failed: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_discord_bot())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
