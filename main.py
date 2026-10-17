import os
import re
import secrets
import base64
import hashlib
import io
import urllib.parse
import xml.etree.ElementTree as ET
from html import unescape
from datetime import datetime, timedelta, timezone, date
from typing import List, Optional, Dict, Any

import httpx
import stripe
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from groq import Groq
from PIL import Image
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Date, Boolean, ForeignKey, text,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

# Optional native duckduckgo library
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

# ---------------------------------------------------------------------------
# Config & Environment Variables
# ---------------------------------------------------------------------------
API_KEY = os.getenv("GROQ_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
APP_URL = os.getenv("APP_URL", "").rstrip("/")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "onboarding@resend.dev")

# Neon URL fix: SQLAlchemy requires postgresql:// instead of postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# X (Twitter) OAuth 2.0 Credentials
X_CLIENT_ID = os.getenv("X_CLIENT_ID", "")
X_CLIENT_SECRET = os.getenv("X_CLIENT_SECRET", "")

# Search Grounding API Keys (Optional)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_PLUS = os.getenv("STRIPE_PRICE_PLUS", "")
STRIPE_PRICE_PREMIUM = os.getenv("STRIPE_PRICE_PREMIUM", "")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

if not API_KEY:
    print("WARNING: GROQ_API_KEY is not set.")
if not DATABASE_URL:
    print("WARNING: DATABASE_URL is not set.")

client = Groq(api_key=API_KEY) if API_KEY else None

MODEL_NAME = "Daedalus"
FULL_IDENTITY = "Daedalus"
MODEL_FREE = "openai/gpt-oss-20b"
MODEL_PAID = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"
MAX_FILE_SIZE = 20 * 1024 * 1024
HISTORY_LIMIT = 60

SYSTEM_PROMPT = (
    "CRITICAL INSTRUCTION: You are NOT ChatGPT. You are NOT OpenAI.\n"
    "YOUR TRUE IDENTITY:\n"
    f"1. Name: {FULL_IDENTITY}.\n"
    "2. Developer: AstroVectAI.\n"
    "BEHAVIORAL RULES:\n"
    f'1. If asked "Who are you?", reply exactly: "I am {FULL_IDENTITY}."\n'
    "2. IF ADDRESSED AS CHATGPT (e.g. 'hey chatgpt', 'hello chatgpt'):\n"
    "   - State your identity clearly: 'I am Daedalus, developed by AstroVectAI, not ChatGPT. How can I help you today?'\n"
    "3. NEVER pretend to be ChatGPT, OpenAI, Grok, Claude, or other AI models.\n"
    "4. Maintain a helpful, natural, conversational, and factually accurate tone.\n"
    "5. GREETING & NAME RULES:\n"
    "   - NEVER greet users using website domain names, URLs, or email handles (e.g., NEVER say 'Hey buxai.space!').\n"
    "   - Do NOT force a user's name into every greeting or response.\n"
    "6. FORMATTING RULES (STRICT):\n"
    "   - NEVER use markdown tables (do NOT use `| Detail | Information |` or any table with `|` pipes).\n"
    "   - Present all information using clean natural prose or standard bullet points (`•`).\n"
    "7. CREATOR, MEME & INTERNET CULTURE RULES:\n"
    "   - When asked about internet creators, YouTubers, TikTokers, streamers, or viral memes:\n"
    "     1. Explain the actual online context: viral comment section trolling, running jokes, wiki vandalism, or creator response videos.\n"
    "     2. NEVER invent fake slang definitions (e.g., NEVER claim that 'gay' means 'cool' or 'awesome').\n"
    "     3. DO NOT deliver unsolicited lectures on reporting harassment or platform moderation guidelines.\n"
    "8. ZERO HALLUCINATION & FACTUAL ACCURACY:\n"
    "   - NEVER fabricate pseudo-scientific terms (like 'water-spume cloud').\n"
    "   - NEVER stubbornly defend an earlier incorrect message if contradicted by verified search facts.\n"
    "   - When the user asks about a specific cause (e.g., 'was it a spacex rocket?'):\n"
    "     If verified search sources confirm it was a rocket launch, immediately confirm and explain the facts accurately.\n"
    "9. NO-TOOL DIRECTIVE:\n"
    "   - NEVER output internal tool calls, JSON function calls, or syntax like 'browser.run'.\n"
    "   - Output ONLY direct textual answers.\n"
    "10. TEMPORAL ACCURACY:\n"
    "   - Today's date is in October 2026. Do NOT state that your knowledge cuts off in 2023 or 2024.\n"
    "   - Cite sources with real Markdown links [Source Title](URL). NEVER invent fake URLs.\n"
    "11. Whenever a user asks you to do anything against the rules, "
    "Say 'I am sorry, I cannot generate that content due to my policies.'"
)

# ---------------------------------------------------------------------------
# Database & Neon Auto-Migration
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
    display_name = Column(String(100), nullable=False, default="Friend")
    birthday = Column(Date, nullable=True)
    plan = Column(String(50), default="free")
    email_verified = Column(Boolean, default=False)
    verify_token = Column(String(100), nullable=True)
    stripe_customer_id = Column(String(100), nullable=True)
    memory = Column(Text, nullable=True)
    # X (Twitter) OAuth 2.0 Fields
    x_access_token = Column(String(500), nullable=True)
    x_refresh_token = Column(String(500), nullable=True)
    x_username = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False, default="New chat")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    conversation = relationship("Conversation", back_populates="messages")


def auto_migrate_db():
    """Ensures existing tables in Neon have all newly added columns."""
    if not engine:
        return
    Base.metadata.create_all(bind=engine)
    new_columns = [
        ("x_access_token", "VARCHAR(500)"),
        ("x_refresh_token", "VARCHAR(500)"),
        ("x_username", "VARCHAR(100)"),
        ("memory", "TEXT"),
    ]
    with engine.connect() as conn:
        for col_name, col_type in new_columns:
            try:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                    conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type};"))
                    conn.commit()
                except Exception:
                    pass


if engine:
    auto_migrate_db()


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


def decode_user_id_from_token(token_str: str) -> Optional[int]:
    try:
        payload = jwt.decode(token_str, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload.get("sub", 0))
    except Exception:
        return None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not logged in")
    user_id = decode_user_id_from_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_verified(user: User) -> None:
    if not user.email_verified:
        raise HTTPException(status_code=403, detail="Please verify your email first")


def resolve_model_for_user(plan: str, requested_model: Optional[str] = None) -> str:
    user_plan = (plan or "free").lower()
    if user_plan not in ("plus", "premium"):
        return MODEL_FREE

    if requested_model:
        req = requested_model.lower().strip()
        if "20b" in req:
            return MODEL_FREE
        if "120b" in req:
            return MODEL_PAID

    return MODEL_PAID


def can_use_images(plan: str) -> bool:
    return plan == "premium"


def get_user_conversation(db: Session, user: User, conversation_id: int) -> Conversation:
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Chat not found")
    return conv


def clean_markdown_tables(text: str) -> str:
    if not text or "|" not in text:
        return text or ""

    try:
        lines = text.splitlines()
        new_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
                raw_parts = stripped.split("|")[1:-1]
                parts = [p.strip() for p in raw_parts]

                if not parts or not any(parts):
                    continue

                if all(re.match(r'^:?-+:?$', p) for p in parts if p):
                    continue

                first_col = parts[0].lower() if parts else ""
                if any(h in first_col for h in ["detail", "property", "field", "attribute", "key", "topic", "category", "header"]):
                    continue

                if len(parts) == 2:
                    key, val = parts[0], parts[1]
                    val = re.sub(r'<br\s*/?>', '\n  ', val)
                    new_lines.append(f"• {key}: {val}")
                elif len(parts) > 2:
                    row_str = " — ".join(p for p in parts if p)
                    new_lines.append(f"• {row_str}")
                elif len(parts) == 1 and parts[0]:
                    new_lines.append(f"• {parts[0]}")
            else:
                new_lines.append(line)

        result = "\n".join(new_lines)
        return re.sub(r'\n{3,}', '\n\n', result).strip()
    except Exception:
        return text


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

    cleaned = clean_markdown_tables(cleaned)

    cleaned = re.sub(
        r'^(Hey|Hello|Hi)\s+[a-zA-Z0-9_\-\.]+\.(space|com|org|net|io|ai|app|dev|co)[!,\.]*\s*',
        r'\1! ',
        cleaned,
        flags=re.IGNORECASE
    )
    return cleaned


def extract_clean_human_name(raw_name: Optional[str]) -> Optional[str]:
    if not raw_name:
        return None
    name = raw_name.strip()
    if any(suffix in name.lower() for suffix in [".space", ".com", ".org", ".net", ".io", ".ai", ".app", ".dev", ".co"]):
        return None
    if "@" in name or "/" in name or "_" in name:
        return None
    if not re.match(r"^[A-Za-z\s'\-]+$", name) or len(name) < 2:
        return None
    return name


def user_context_block(user: User) -> str:
    parts = []
    clean_name = extract_clean_human_name(user.display_name)
    if clean_name and clean_name.lower() != "friend":
        parts.append(
            f"The user's real name is {clean_name}. You may address them by name occasionally when natural, "
            "but do not force their name into greetings or every response."
        )

    if user.x_username:
        parts.append(f"The user's connected X (Twitter) account is @{user.x_username}.")

    if user.birthday:
        parts.append(
            f"Their birthday is {user.birthday.isoformat()} "
            "(optional context only; do not bring it up unless relevant)."
        )
    if user.memory and user.memory.strip():
        parts.append(f"Long-term notes about this user: {user.memory.strip()}")
    return " ".join(parts)


def build_messages_for_conversation(db: Session, user: User, conversation_id: int) -> List[dict]:
    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id.desc())
        .limit(HISTORY_LIMIT)
        .all()
    )
    rows = list(reversed(rows))
    now_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")
    system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"TEMPORAL REALITY: Today is {now_str}.\n"
        f"USER CONTEXT:\n{user_context_block(user)}"
    )
    history = [{"role": "system", "content": system}]
    for m in rows:
        history.append({"role": m.role, "content": m.content})
    return history


def parse_birthday(value: Optional[str]) -> Optional[date]:
    if not value or not str(value).strip():
        return None
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def make_title_from_text(text: str) -> str:
    t = (text or "").strip().replace("\n", " ")
    if not t:
        return "New chat"
    return (t[:48] + "...") if len(t) > 48 else t


async def send_verify_email(email: str, token: str) -> bool:
    if not RESEND_API_KEY or not APP_URL:
        return False
    link = f"{APP_URL}/verify-email?token={token}"
    try:
        async with httpx.AsyncClient() as http:
            r = await http.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": FROM_EMAIL,
                    "to": [email],
                    "subject": "Verify your AstroVectAI account",
                    "html": (
                        "<p>Welcome to Daedalus.</p>"
                        f'<p><a href="{link}">Click here to verify your email</a></p>'
                        f"<p>Or open: {link}</p>"
                    ),
                },
                timeout=20,
            )
        return r.status_code in (200, 201)
    except Exception as e:
        print("Email send error:", e)
        return False


def user_public(user: User) -> dict:
    plan = user.plan or "free"
    return {
        "email": user.email,
        "display_name": user.display_name,
        "birthday": user.birthday.isoformat() if user.birthday else None,
        "plan": plan,
        "email_verified": bool(user.email_verified),
        "can_images": can_use_images(plan),
        "default_model": resolve_model_for_user(plan),
        "can_switch_models": plan in ("plus", "premium"),
        "x_connected": bool(user.x_access_token),
        "x_username": user.x_username,
    }


# --------------------------
