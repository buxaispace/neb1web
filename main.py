import os
import base64
import io
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from groq import Groq
from PIL import Image
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.getenv("GROQ_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

if not API_KEY:
    print("WARNING: GROQ_API_KEY environment variable is not set.")
if not DATABASE_URL:
    print("WARNING: DATABASE_URL is not set.")

client = Groq(api_key=API_KEY) if API_KEY else None

MODEL_NAME = "Nebula 1"
FULL_IDENTITY = "Nebula 1"
CHAT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"
MAX_FILE_SIZE = 20 * 1024 * 1024

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

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    plan = Column(String(50), default="free")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user = relationship("User", back_populates="messages")


if engine:
    Base.metadata.create_all(bind=engine)


def get_db():
    if not SessionLocal:
        raise HTTPException(status_code=500, detail="Database not configured")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": str(user_id), "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub", 0))
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


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


def build_messages_for_user(db: Session, user_id: int) -> List[dict]:
    rows = (
        db.query(Message)
        .filter(Message.user_id == user_id)
        .order_by(Message.id.desc())
        .limit(40)
        .all()
    )
    rows = list(reversed(rows))
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in rows:
        history.append({"role": m.role, "content": m.content})
    return history


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Nebula 1")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_name": MODEL_NAME})


# ---------- Auth ----------
@app.post("/register")
async def register(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    email = email.strip().lower()
    if not email or "@" not in email:
        return JSONResponse({"error": "Invalid email"}, status_code=400)
    if len(password) < 6:
        return JSONResponse({"error": "Password must be at least 6 characters"}, status_code=400)
    if db.query(User).filter(User.email == email).first():
        return JSONResponse({"error": "Email already registered"}, status_code=400)

    user = User(email=email, password_hash=hash_password(password), plan="free")
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id)
    return JSONResponse({"token": token, "email": user.email, "plan": user.plan})


@app.post("/login")
async def login(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    email = email.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return JSONResponse({"error": "Wrong email or password"}, status_code=401)
    token = create_access_token(user.id)
    return JSONResponse({"token": token, "email": user.email, "plan": user.plan})


@app.get("/me")
async def me(user: User = Depends(get_current_user)):
    return JSONResponse({"email": user.email, "plan": user.plan})


@app.get("/history")
async def history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(Message)
        .filter(Message.user_id == user.id)
        .order_by(Message.id.asc())
        .limit(100)
        .all()
    )
    return JSONResponse({
        "messages": [{"role": m.role, "content": m.content} for m in rows]
    })


# ---------- Chat ----------
@app.post("/chat")
async def chat(
    message: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not client:
        return JSONResponse({"error": "Server missing GROQ_API_KEY"}, status_code=500)

    user_text = message.strip()
    if not user_text:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    db.add(Message(user_id=user.id, role="user", content=user_text))
    db.commit()

    messages = build_messages_for_user(db, user.id)

    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.5,
        )
        raw = completion.choices[0].message.content
        final = sanitize_response(raw)
        db.add(Message(user_id=user.id, role="assistant", content=final))
        db.commit()
        return JSONResponse({"reply": final})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not client:
        return JSONResponse({"error": "Server missing GROQ_API_KEY"}, status_code=500)

    content = b""
    chunk_size = 1024 * 1024

    try:
        while chunk := await file.read(chunk_size):
            content += chunk
            if len(content) > MAX_FILE_SIZE:
                return JSONResponse({"error": "Image too large (max 20 MB)"}, status_code=400)
    finally:
        await file.close()

    try:
        img = Image.open(io.BytesIO(content))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail((2048, 2048))
        compressed_buffer = io.BytesIO()
        img.save(compressed_buffer, format="JPEG", quality=85, optimize=True)
        content = compressed_buffer.getvalue()
        mime = "image/jpeg"
    except Exception as e:
        return JSONResponse({"error": f"Failed to process or compress image: {e}"}, status_code=400)

    b64 = base64.b64encode(content).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

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

    note = f"[User uploaded an image]\nImage description: {image_description}"
    if caption and caption.strip():
        note += f"\nUser said: {caption.strip()}"

    db.add(Message(user_id=user.id, role="user", content=note))
    db.commit()

    messages = build_messages_for_user(db, user.id)

    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.5,
        )
        raw = completion.choices[0].message.content
        final = sanitize_response(raw)
        db.add(Message(user_id=user.id, role="assistant", content=final))
        db.commit()
        return JSONResponse({
            "reply": final,
            "filename": file.filename,
            "size": len(content),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/reset")
async def reset(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(Message).filter(Message.user_id == user.id).delete()
    db.commit()
    return JSONResponse({"status": "ok"})
