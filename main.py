import os
import re
import secrets
import base64
import io
import imaplib
import email
from email.header import decode_header
import asyncio
import urllib.parse
import xml.etree.ElementTree as ET
from html import unescape
from datetime import datetime, timedelta, timezone, date
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
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

# ---------------------------------------------------------------------------
# App Initialization (Declared at top so Uvicorn always finds "app")
# ---------------------------------------------------------------------------
app = FastAPI(title="Daedalus")
templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Cash App & Automated Email Receipt Verification Config
CASHAPP_TAG = os.getenv("CASHAPP_TAG", "").strip()
EMAIL_USER = os.getenv("EMAIL_USER", "").strip()
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "").replace(" ", "").strip()
EMAIL_IMAP_HOST = os.getenv("EMAIL_IMAP_HOST", "imap.gmail.com").strip()

# Search Grounding API Keys (Optional)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

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
    "   - If the user asks what their username, display name, or preferred name is, tell them their name from USER CONTEXT.\n"
    "6. FORMATTING RULES (STRICT):\n"
    "   - NEVER use markdown tables (do NOT use `| Detail | Information |` or any table with `|` pipes).\n"
    "   - Present non-code information using clean natural prose or standard bullet points (`•`).\n"
    "   - For code, ALWAYS use properly fenced markdown code blocks (e.g. ```python, ```html, ```javascript) with complete, runnable code.\n"
    "7. CREATOR, MEME & INTERNET CULTURE RULES:\n"
    "   - When asked about internet creators, YouTubers, TikTokers, streamers, or viral memes:\n"
    "     1. Explain the actual online context: viral comment section trolling, running jokes, wiki vandalism, or creator response videos.\n"
    "     2. NEVER invent fake slang definitions (e.g., NEVER claim that 'gay' means 'cool' or 'awesome').\n"
    "     3. DO NOT deliver unsolicited lectures on reporting harassment or platform moderation guidelines.\n"
    "8. ZERO HALLUCINATION & FACTUAL ACCURACY:\n"
    "   - NEVER fabricate pseudo-scientific terms (like 'water-spume cloud').\n"
    "   - NEVER stubbornly defend an earlier incorrect message if contradicted by verified search facts.\n"
    "9. NO-TOOL DIRECTIVE:\n"
    "   - NEVER output internal tool calls, JSON function calls, or syntax like 'browser.run'.\n"
    "   - Output ONLY direct textual answers.\n"
    "10. TEMPORAL ACCURACY:\n"
    "   - Today's date is in October 2026. Do NOT state that your knowledge cuts off in 2023 or 2024.\n"
    "   - Cite sources with real Markdown links [Source Title](URL). NEVER invent fake URLs.\n"
    "11. REFUSAL & FRESH-TURN POLICY (CRITICAL):\n"
    "   - Evaluate every new user message independently.\n"
    "   - If a previous message in the conversation was refused, DO NOT repeat the refusal on the next message.\n"
    "   - Treat every new message as a fresh turn and answer normally unless the new message itself directly asks for severe harm or illegal acts."
)

CODING_EXPERT_DIRECTIVE = (
    "PRINCIPAL SOFTWARE ENGINEER & CODING MODE ACTIVE:\n"
    "- Write complete, production-ready, bug-free, and directly runnable code.\n"
    "- NEVER use lazy placeholders like '// ... rest of code here' or '# TODO: implement'. Write the full implementation.\n"
    "- Wrap all code in fenced markdown blocks with the language tag (e.g. ```python, ```javascript, ```html, ```lua).\n"
    "- Double-check all imports, variable names, edge cases, and syntax before outputting."
)

# ---------------------------------------------------------------------------
# Database & Models
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
    payment_code = Column(String(100), nullable=True)
    pending_plan = Column(String(50), nullable=True)
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
    """Ensures existing tables in Neon have all required columns."""
    if not engine:
        return
    try:
        Base.metadata.create_all(bind=engine)
        cols = [
            ("memory", "TEXT"),
            ("payment_code", "VARCHAR(100)"),
            ("pending_plan", "VARCHAR(50)"),
        ]
        with engine.connect() as conn:
            for col_name, col_type in cols:
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
    except Exception as e:
        print(f"Neon database migration notice: {e}")


@app.on_event("startup")
def on_startup():
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
) -> User:
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


# ---------------------------------------------------------------------------
# TIER CAPABILITIES & REASONING LEVEL ENFORCEMENT
# ---------------------------------------------------------------------------
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


def get_reasoning_effort_for_plan(plan: str) -> str:
    """
    Free -> locked to 'low'
    Plus -> locked to 'medium'
    Premium -> unlocked to 'high'
    """
    user_plan = (plan or "free").lower()
    if user_plan == "premium":
        return "high"
    if user_plan == "plus":
        return "medium"
    return "low"


def get_max_tokens_for_plan(plan: str) -> int:
    user_plan = (plan or "free").lower()
    if user_plan == "premium":
        return 8192
    if user_plan == "plus":
        return 4096
    return 2048


def can_use_images(plan: str) -> bool:
    """Both Plus and Premium users get Vision-to-Code & Screenshot Bug Fixer."""
    return (plan or "free").lower() in ("plus", "premium")


def get_user_conversation(db: Session, user: User, conversation_id: int) -> Conversation:
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Chat not found")
    return conv


def clean_markdown_tables_outside_code(text_segment: str) -> str:
    if not text_segment or "|" not in text_segment:
        return text_segment or ""

    try:
        lines = text_segment.splitlines()
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
        return re.sub(r'\n{3,}', '\n\n', result)
    except Exception:
        return text_segment


def clean_markdown_tables(text: str) -> str:
    """Protects all fenced ```code``` blocks while converting markdown tables outside of code into bullets."""
    if not text or "|" not in text:
        return text or ""
    parts = re.split(r"(```[\s\S]*?```)", text)
    out = []
    for part in parts:
        if part.startswith("```") and part.endswith("```"):
            out.append(part)
        else:
            out.append(clean_markdown_tables_outside_code(part))
    return "".join(out).strip()


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


def is_slur_or_restricted_name(raw_name: Optional[str]) -> bool:
    if not raw_name:
        return False
    lower = raw_name.lower().strip()
    restricted_patterns = [
        r"n+[i1l!]+g+[g]+[e3a@4]+r*",
        r"n+[i1l!]+g+[a@4]+",
        r"f+[a@4]+g+[g]*[o0]+t*",
        r"k+[i1]+k+[e3]+",
    ]
    return any(re.search(pat, lower) for pat in restricted_patterns)


def extract_clean_display_name(raw_name: Optional[str]) -> Optional[str]:
    if not raw_name:
        return None
    name = raw_name.strip()
    if any(suffix in name.lower() for suffix in [".space", ".com", ".org", ".net", ".io", ".ai", ".app", ".dev", ".co"]):
        return None
    if "@" in name or "/" in name:
        return None
    if len(name) < 1 or len(name) > 100:
        return None
    return name


def user_context_block(user: User) -> str:
    parts = []
    if is_slur_or_restricted_name(user.display_name):
        parts.append(
            "The user's display name in settings is currently set to an offensive/restricted word. "
            "Do NOT repeat the slur, and do NOT refuse normal messages—if they ask what their name/username is, "
            "politely let them know their display name is set to a restricted word and they can change it using the edit button in the header."
        )
    else:
        clean_name = extract_clean_display_name(user.display_name)
        if clean_name:
            parts.append(
                f"The user's display name / username is '{clean_name}'. "
                f"If the user asks 'what is my username?' or 'what name did I set for you to call me?', "
                f"tell them directly that their name is '{clean_name}'."
            )

    if user.birthday:
        parts.append(
            f"Their birthday is {user.birthday.isoformat()} "
            "(optional context only; do not bring it up unless relevant)."
        )
    if user.memory and user.memory.strip():
        parts.append(f"Long-term notes & preferences about this user: {user.memory.strip()}")
    return " ".join(parts)


def is_refusal_message(content: str) -> bool:
    if not content:
        return False
    lower = content.lower().strip()
    refusal_phrases = [
        "i'm sorry, but i can't comply with that",
        "i’m sorry, but i can’t comply with that",
        "i cannot comply with that",
        "i'm sorry, but i can't help with that",
        "i’m sorry, but i can’t help with that",
        "i'm sorry, i can't provide that information",
        "i’m sorry, i can’t provide that information",
        "i'm sorry, but i don't have that information",
        "i’m sorry, but i don’t have that information",
        "i am sorry, i cannot generate that content due to my policies",
        "i'm sorry, i can't generate that content",
    ]
    return len(lower) < 180 and any(p in lower for p in refusal_phrases)


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
    reasoning_level = get_reasoning_effort_for_plan(user.plan or "free")
    system = (
        f"Reasoning: {reasoning_level}\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"TEMPORAL REALITY: Today is {now_str}.\n"
        f"USER CONTEXT:\n{user_context_block(user)}"
    )
    history = [{"role": "system", "content": system}]

    cleaned_rows = []
    i = 0
    while i < len(rows):
        m = rows[i]
        if m.role == "user" and (i + 1) < len(rows) and rows[i + 1].role == "assistant":
            if is_refusal_message(rows[i + 1].content):
                i += 2
                continue
        if m.role == "assistant" and is_refusal_message(m.content):
            i += 1
            continue
        cleaned_rows.append(m)
        i += 1

    for m in cleaned_rows:
        history.append({"role": m.role, "content": m.content})
    return history


def is_coding_or_logic_query(text: str) -> bool:
    """Detects software engineering, coding, debugging, math, or algorithm prompts."""
    lower = text.lower()
    if "```" in text:
        return True
    coding_keywords = [
        "code", "script", "python", "javascript", "typescript", "html", "css",
        "react", "nextjs", "fastapi", "flask", "django", "sql", "postgres",
        "luau", "roblox", "unity", "c#", "c++", "java", "rust", "golang",
        "function", "variable", "bug", "error", "traceback", "exception",
        "syntax", "compile", "algorithm", "regex", "json", "api endpoint",
        "write a program", "make a website", "build a game", "debug", "fix this"
    ]
    return any(k in lower for k in coding_keywords)


def maybe_update_user_memory(db: Session, user: User, user_text: str):
    """Automatically extracts persistent user facts/preferences for Plus & Premium users."""
    if (user.plan or "free").lower() not in ("plus", "premium") or not client:
        return
    lower = user_text.lower()
    memory_triggers = [
        "remember that", "my project is", "i am building", "i'm building",
        "i code in", "my favorite language", "i prefer", "i use ", "my stack is"
    ]
    if not any(t in lower for t in memory_triggers):
        return
    try:
        current_mem = (user.memory or "").strip()
        resp = client.chat.completions.create(
            model=MODEL_FREE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract 1 brief, factual note about the user's preferences, tech stack, or projects "
                        "from their message to append to their long-term memory. Output ONLY the short note (max 20 words)."
                    )
                },
                {"role": "user", "content": user_text}
            ],
            temperature=0.1,
            max_tokens=40,
        )
        new_note = (resp.choices[0].message.content or "").strip()
        if new_note and new_note not in current_mem:
            updated = (current_mem + " | " + new_note).strip(" |")
            user.memory = updated[:4000]
            db.commit()
    except Exception as e:
        print(f"Auto-memory update notice: {e}")


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


async def send_verify_email(email_addr: str, token: str) -> bool:
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
                    "to": [email_addr],
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
        "reasoning_effort": get_reasoning_effort_for_plan(plan),
    }


# ---------------------------------------------------------------------------
# AUTOMATED CASH APP EMAIL RECEIPT VERIFICATION ENGINE
# ---------------------------------------------------------------------------
def decode_mime_Text(val: Optional[str]) -> str:
    if not val:
        return ""
    decoded_parts = decode_header(val)
    out = []
    for part, enc in decoded_parts:
        if isinstance(part, bytes):
            out.append(part.decode(enc or "utf-8", errors="ignore"))
        else:
            out.append(str(part))
    return "".join(out)


def extract_email_body(msg: email.message.Message) -> str:
    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdisp = str(part.get("Content-Disposition") or "")
            if ctype in ("text/plain", "text/html") and "attachment" not in cdisp:
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        text_chunk = payload.decode(charset, errors="ignore")
                        if ctype == "text/html":
                            text_chunk = re.sub(r"<[^>]+>", " ", text_chunk)
                        body_parts.append(unescape(text_chunk))
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                text_chunk = payload.decode(charset, errors="ignore")
                text_chunk = re.sub(r"<[^>]+>", " ", text_chunk)
                body_parts.append(unescape(text_chunk))
        except Exception:
            pass
    return " ".join(body_parts)


def check_imap_for_cashapp_receipt(payment_code: str, expected_plan: str) -> Optional[str]:
    if not EMAIL_USER or not EMAIL_APP_PASSWORD or not payment_code:
        return None

    try:
        mail = imaplib.IMAP4_SSL(EMAIL_IMAP_HOST, 993)
        mail.login(EMAIL_USER, EMAIL_APP_PASSWORD)
        mail.select("INBOX")

        status, data = mail.search(None, "ALL")
        if status != "OK" or not data or not data[0]:
            mail.logout()
            return None

        msg_ids = data[0].split()
        recent_ids = list(reversed(msg_ids[-30:]))

        code_lower = payment_code.lower().strip()

        for m_id in recent_ids:
            res, msg_data = mail.fetch(m_id, "(RFC822)")
            if res != "OK" or not msg_data:
                continue

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    from_header = decode_mime_Text(msg.get("From", "")).lower()
                    subject_header = decode_mime_Text(msg.get("Subject", ""))
                    body_text = extract_email_body(msg)

                    combined_text = f"{subject_header} {body_text}"
                    combined_lower = combined_text.lower()

                    is_from_cashapp = any(
                        domain in from_header
                        for domain in ["square.com", "cash.app", "squareup.com"]
                    )
                    if not is_from_cashapp:
                        continue

                    if code_lower not in combined_lower:
                        continue

                    if not any(w in combined_lower for w in ["sent you", "received", "paid you", "you got"]):
                        continue

                    if re.search(r"\$\s*(?:8|9|1\d)(?:\.00)?\b", combined_text):
                        mail.logout()
                        return "premium"
                    if re.search(r"\$\s*[4567](?:\.00)?\b", combined_text):
                        mail.logout()
                        return "plus"

        mail.logout()
    except Exception as e:
        print(f"Cash App IMAP check error: {e}")

    return None


# ---------------------------------------------------------------------------
# SOCIAL MEDIA, TIKTOK OEMBED & CREATOR SEARCH ENGINE
# ---------------------------------------------------------------------------
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def extract_urls(text: str) -> List[str]:
    return re.findall(r'https?://[^\s<>"\')]+', text, flags=re.IGNORECASE)


def clean_html(html: str, max_chars: int = 3000) -> str:
    if not html:
        return ""
    text = re.sub(r'<(script|style|svg|noscript|header|footer|nav|aside|form)[^>]*>.*?</\1>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = unescape(text)
    return re.sub(r'\s+', ' ', text).strip()[:max_chars]


async def fetch_tiktok_video_data(url: str) -> Optional[Dict[str, str]]:
    try:
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True, headers=BROWSER_HEADERS) as http:
            oembed_url = f"https://www.tiktok.com/oembed?url={urllib.parse.quote(url)}"
            res = await http.get(oembed_url)
            if res.status_code == 200:
                data = res.json()
                title = data.get("title", "")
                author = data.get("author_name", "")
                handle = data.get("author_unique_id", "")
                content = f"TikTok Video by @{handle} ({author}):\nCaption / Description: {title}"
                return {
                    "title": f"TikTok: @{handle} - {title[:60]}",
                    "url": url,
                    "snippet": content[:280],
                    "content": content,
                    "site": "tiktok.com"
                }
    except Exception as e:
        print("TikTok oEmbed warning:", e)
    return None


async def fetch_webpage_content(url: str, max_chars: int = 2500) -> Optional[Dict[str, str]]:
    if "tiktok.com" in url:
        tiktok_data = await fetch_tiktok_video_data(url)
        if tiktok_data:
            return tiktok_data

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, headers=BROWSER_HEADERS) as http:
            res = await http.get(url)
            if res.status_code == 200:
                html = res.text

                if "news.google.com" in url:
                    dest_match = re.search(r'href=[\'"](https?://(?!news\.google\.com)[^\'"]+)[\'"]', html)
                    if dest_match:
                        real_url = dest_match.group(1)
                        pub_res = await http.get(real_url)
                        if pub_res.status_code == 200:
                            html = pub_res.text
                            url = real_url

                title_match = re.search(r'<title[^>]*>(.*?)</title>', html, flags=re.IGNORECASE | re.DOTALL)
                title = unescape(title_match.group(1)).strip() if title_match else url
                body = clean_html(html, max_chars)
                if body and len(body) > 100:
                    return {
                        "title": title[:100],
                        "url": url,
                        "snippet": body[:280] + "...",
                        "content": body,
                        "site": urllib.parse.urlparse(url).netloc,
                    }
    except Exception as e:
        print(f"Error reading {url}: {e}")
    return None


def is_personal_or_account_query(text: str) -> bool:
    lower = text.lower().strip()
    personal_patterns = [
        "my username", "my name", "call me", "who am i", "what about now",
        "updated now", "my birthday", "my account", "my plan", "who are you",
        "your name", "what is your name"
    ]
    if any(p in lower for p in personal_patterns):
        return True
    if len(lower.split()) <= 2 and not extract_urls(text):
        return True
    return False


def generate_smart_search_queries(messages: List[dict], user_text: str) -> List[str]:
    user_turns = [m["content"] for m in messages if m.get("role") == "user"]
    queries = []

    combined = " ".join(user_turns[-3:])
    cleaned = re.sub(
        r'\b(why not|why|what happened to|tell me about|who is|can you|the one with|that|back in|can you help|explain)\b',
        ' ',
        combined,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if len(cleaned) > 3:
        queries.append(cleaned)

    lower_u = user_text.lower()
    if any(k in lower_u for k in ["why is everyone calling", "meme", "gay", "drama", "tiktok", "youtuber", "creator"]):
        queries.append(f"{cleaned} tiktok meme")
        queries.append(f"{cleaned} reddit drama")

    if client and len(user_turns) >= 1:
        try:
            history_snips = [f"{m['role']}: {m['content'][:120]}" for m in messages[-4:] if m.get("role") in ("user", "assistant")]
            exp_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert search query formulation engine. Given the user dialogue, output 2 specific 3-to-5-word "
                        "search queries that will locate the exact content creator discussion, technical docs, or news story being asked about. "
                        "Output each query on a new line with no quotes or numbers."
                    )
                },
                {"role": "user", "content": "Context:\n" + "\n".join(history_snips) + f"\nLatest: {user_text}"}
            ]
            resp = client.chat.completions.create(
                model=MODEL_FREE,
                messages=exp_prompt,
                temperature=0.1,
                max_tokens=40,
            )
            for line in resp.choices[0].message.content.splitlines():
                clean_line = line.strip().strip('"').strip("'")
                if clean_line and clean_line not in queries:
                    queries.append(clean_line)
        except Exception:
            pass

    return queries[:3] if queries else [user_text]


async def query_reddit_search(query: str) -> List[Dict[str, str]]:
    results = []
    try:
        reddit_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        q_enc = urllib.parse.quote_plus(query)
        url = f"https://www.reddit.com/search.json?q={q_enc}&sort=relevance&limit=4"
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True, headers=reddit_headers) as http:
            r = await http.get(url)
            if r.status_code == 200:
                data = r.json()
                for child in data.get("data", {}).get("children", [])[:4]:
                    d = child.get("data", {})
                    title = unescape(d.get("title", ""))
                    selftext = unescape(clean_html(d.get("selftext", ""), 400))
                    sub = d.get("subreddit_name_prefixed", "r/Reddit")
                    permalink = d.get("permalink", "")
                    post_url = f"https://www.reddit.com{permalink}" if permalink else d.get("url", "")
                    body_text = f"Discussion on {sub}: {title}\nPost context: {selftext}"
                    results.append({
                        "title": f"Reddit ({sub}): {title[:80]}",
                        "url": post_url,
                        "snippet": body_text[:280],
                        "content": body_text,
                        "site": "reddit.com"
                    })
    except Exception as e:
        print("Reddit search warning:", e)
    return results


async def query_google_news_rss(query: str) -> List[Dict[str, str]]:
    results = []
    try:
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True, headers=BROWSER_HEADERS) as http:
            q_enc = urllib.parse.quote_plus(query)
            rss_url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-US&gl=US&ceid=US:en"
            r = await http.get(rss_url)
            if r.status_code == 200:
                root = ET.fromstring(r.text)
                for item in root.findall(".//item")[:4]:
                    title = unescape(re.sub(r'<[^>]+>', '', item.findtext("title") or "")).strip()
                    link = (item.findtext("link") or "").strip()
                    desc = unescape(re.sub(r'<[^>]+>', '', item.findtext("description") or "")).strip()
                    source = item.findtext("source") or "News"
                    if title and link:
                        results.append({
                            "title": title,
                            "url": link,
                            "snippet": desc[:280],
                            "site": source
                        })
    except Exception as e:
        print("Google News RSS warning:", e)
    return results


async def execute_python_rag(query: str, max_sources: int = 4, include_news: bool = True) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []

    # 1. Reddit Social / Technical / Creator Search
    reddit_res = await query_reddit_search(query)
    if reddit_res:
        sources.extend(reddit_res)

    # 2. Google News RSS (Skipped on pure coding queries so news doesn't pollute code)
    if include_news:
        news_res = await query_google_news_rss(query)
        if news_res:
            sources.extend(news_res)

    # 3. Tavily API
    if TAVILY_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=6.0) as http:
                r = await http.post(
                    "https://api.tavily.com/search",
                    json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_sources},
                )
                if r.status_code == 200:
                    for item in r.json().get("results", []):
                        sources.append({
                            "title": item.get("title", "Source"),
                            "url": item.get("url", ""),
                            "snippet": item.get("content", ""),
                            "site": urllib.parse.urlparse(item.get("url", "")).netloc,
                        })
        except Exception as e:
            print("Tavily warning:", e)

    # 4. Serper API
    if SERPER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=6.0) as http:
                r = await http.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={"q": query, "num": max_sources},
                )
                if r.status_code == 200:
                    for item in r.json().get("organic", [])[:max_sources]:
                        sources.append({
                            "title": item.get("title", "Google Result"),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "site": urllib.parse.urlparse(item.get("link", "")).netloc,
                        })
        except Exception as e:
            print("Serper warning:", e)

    # 5. Native DDGS
    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_sources):
                    href = r.get("href", "")
                    if href.startswith("http"):
                        sources.append({
                            "title": r.get("title", "Web Result"),
                            "url": href,
                            "snippet": r.get("body", ""),
                            "site": urllib.parse.urlparse(href).netloc,
                        })
        except Exception as e:
            print("DDGS warning:", e)

    # 6. Standalone DuckDuckGo Lite HTTP Search
    try:
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True, headers=BROWSER_HEADERS) as http:
            res = await http.post("https://lite.duckduckgo.com/lite/", data={"q": query})
            if res.status_code == 200:
                html_text = res.text
                link_pattern = re.compile(r'<a[^>]+class=[\'"]result-link[\'"][^>]+href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
                snippet_pattern = re.compile(r'<td[^>]+class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL)
                links = link_pattern.findall(html_text)
                snippets = snippet_pattern.findall(html_text)
                for i in range(min(len(links), len(snippets), max_sources)):
                    raw_url, raw_title = links[i]
                    if "uddg=" in raw_url:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                        url = parsed.get("uddg", [raw_url])[0]
                    else:
                        url = raw_url
                    clean_title = unescape(re.sub(r'<[^>]+>', '', raw_title).strip())
                    clean_snippet = unescape(re.sub(r'<[^>]+>', '', snippets[i]).strip())
                    if url.startswith("http") and clean_title:
                        sources.append({
                            "title": clean_title,
                            "url": url,
                            "snippet": clean_snippet,
                            "site": urllib.parse.urlparse(url).netloc
                        })
    except Exception as e:
        print("DDG Lite warning:", e)

    unique_sources = []
    seen = set()
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            unique_sources.append(s)

    crawl_count = 3 if max_sources >= 6 else 2
    for i in range(min(crawl_count, len(unique_sources))):
        s = unique_sources[i]
        crawled = await fetch_webpage_content(s["url"])
        if crawled and crawled.get("content"):
            s["content"] = crawled["content"]
            s["url"] = crawled["url"]

    return unique_sources[:max_sources]


# ---------------------------------------------------------------------------
# Home Route & User Authentication
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "model_name": MODEL_NAME}
    )


@app.post("/register")
async def register(
    email: str = Form(...),
    password: str = Form(...),
    display_name: str = Form(...),
    birthday: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    try:
        email = email.strip().lower()
        name = display_name.strip()
        if not email or "@" not in email:
            return JSONResponse({"error": "Invalid email"}, status_code=400)
        if len(password) < 6:
            return JSONResponse({"error": "Password must be at least 6 characters"}, status_code=400)
        if not name or len(name) > 100:
            return JSONResponse({"error": "Please enter a name"}, status_code=400)
        if db.query(User).filter(User.email == email).first():
            return JSONResponse({"error": "Email already registered"}, status_code=400)

        bday = parse_birthday(birthday)
        if birthday and str(birthday).strip() and bday is None:
            return JSONResponse({"error": "Birthday must be a valid date"}, status_code=400)

        token = secrets.token_urlsafe(32)
        auto_verify = not (RESEND_API_KEY and APP_URL)

        user = User(
            email=email,
            password_hash=hash_password(password),
            display_name=name,
            birthday=bday,
            plan="free",
            email_verified=auto_verify,
            verify_token=None if auto_verify else token,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        conv = Conversation(user_id=user.id, title="New chat")
        db.add(conv)
        db.commit()
        db.refresh(conv)

        email_sent = False
        if not auto_verify:
            email_sent = await send_verify_email(email, token)

        return JSONResponse({
            "token": create_access_token(user.id),
            **user_public(user),
            "conversation_id": conv.id,
            "needs_verification": not user.email_verified,
            "email_sent": email_sent,
            "message": (
                None if user.email_verified
                else (
                    "Check your email for a verification link."
                    if email_sent
                    else "Verification email could not be sent."
                )
            ),
        })
    except Exception as e:
        print(f"Register error: {e}")
        return JSONResponse({"error": f"Database error: {str(e)}"}, status_code=500)


@app.post("/login")
async def login(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        email = email.strip().lower()
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.password_hash):
            return JSONResponse({"error": "Wrong email or password"}, status_code=401)

        conv = (
            db.query(Conversation)
            .filter(Conversation.user_id == user.id)
            .order_by(Conversation.updated_at.desc())
            .first()
        )
        if not conv:
            conv = Conversation(user_id=user.id, title="New chat")
            db.add(conv)
            db.commit()
            db.refresh(conv)

        return JSONResponse({
            "token": create_access_token(user.id),
            **user_public(user),
            "conversation_id": conv.id,
            "needs_verification": not user.email_verified,
        })
    except Exception as e:
        print(f"Login error: {e}")
        return JSONResponse({"error": f"Database error: {str(e)}"}, status_code=500)


@app.get("/verify-email")
async def verify_email(token: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.verify_token == token).first()
    if not user:
        return HTMLResponse(
            "<h2>Invalid or expired link</h2><p><a href='/'>Back to Daedalus</a></p>",
            status_code=400,
        )
    user.email_verified = True
    user.verify_token = None
    db.commit()
    return HTMLResponse(
        "<h2>Email verified</h2>"
        "<p>You can close this tab and return to Daedalus.</p>"
        "<p><a href='/'>Open Daedalus</a></p>"
    )


@app.post("/resend-verification")
async def resend_verification(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if user.email_verified:
        return JSONResponse({"status": "already_verified"})
    token = secrets.token_urlsafe(32)
    user.verify_token = token
    db.commit()
    if not RESEND_API_KEY or not APP_URL:
        user.email_verified = True
        user.verify_token = None
        db.commit()
        return JSONResponse({"status": "auto_verified", "email_verified": True})
    ok = await send_verify_email(user.email, token)
    if not ok:
        return JSONResponse({"error": "Could not send email"}, status_code=500)
    return JSONResponse({"status": "sent"})


@app.get("/me")
async def me(user: User = Depends(get_current_user)):
    return JSONResponse(user_public(user))


@app.post("/update-profile")
async def update_profile(
    display_name: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    name = display_name.strip()
    if not name or len(name) > 100:
        return JSONResponse({"error": "Name must be 1-100 characters"}, status_code=400)
    user.display_name = name
    db.commit()
    return JSONResponse(user_public(user))


@app.post("/update-memory")
async def update_memory(
    memory: str = Form(""),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user.memory = (memory or "").strip()[:4000] or None
    db.commit()
    return JSONResponse({"memory": user.memory or ""})


@app.get("/conversations")
async def list_conversations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return JSONResponse({
        "conversations": [
            {
                "id": c.id,
                "title": c.title,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            }
            for c in rows
        ]
    })


@app.post("/conversations")
async def create_conversation(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_verified(user)
    conv = Conversation(user_id=user.id, title="New chat")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return JSONResponse({"id": conv.id, "title": conv.title})


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = get_user_conversation(db, user, conversation_id)
    db.delete(conv)
    db.commit()
    remaining = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .first()
    )
    if not remaining:
        remaining = Conversation(user_id=user.id, title="New chat")
        db.add(remaining)
        db.commit()
        db.refresh(remaining)
    return JSONResponse({"status": "ok", "conversation_id": remaining.id})


@app.get("/history")
async def history(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_user_conversation(db, user, conversation_id)
    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id.asc())
        .limit(200)
        .all()
    )
    return JSONResponse({
        "messages": [{"role": m.role, "content": m.content} for m in rows]
    })


# ---------------------------------------------------------------------------
# CHAT ENDPOINT (TIERED REASONING + CODING MODE + 2-PASS PREMIUM CRITIQUE)
# ---------------------------------------------------------------------------
@app.post("/chat")
async def chat(
    message: str = Form(...),
    conversation_id: int = Form(...),
    model: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_verified(user)
    if not client:
        return JSONResponse({"error": "Server missing GROQ_API_KEY"}, status_code=500)

    try:
        db.refresh(user)

        conv = get_user_conversation(db, user, conversation_id)
        user_text = message.strip()
        if not user_text:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        msg_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
        if msg_count == 0 and conv.title == "New chat":
            conv.title = make_title_from_text(user_text)

        db.add(Message(conversation_id=conv.id, role="user", content=user_text))
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()

        # Auto-extract user/project memory for Plus & Premium users
        maybe_update_user_memory(db, user, user_text)

        user_plan = (user.plan or "free").lower()
        active_model = resolve_model_for_user(user_plan, model)
        max_out_tokens = get_max_tokens_for_plan(user_plan)

        messages = build_messages_for_conversation(db, user, conv.id)
        lowered_text = user_text.lower().strip()

        if lowered_text in ["hey chatgpt", "hello chatgpt", "hi chatgpt", "chatgpt"]:
            final = "I am Daedalus, developed by AstroVectAI, not ChatGPT. How can I help you today?"
            db.add(Message(conversation_id=conv.id, role="assistant", content=final))
            conv.updated_at = datetime.now(timezone.utc)
            db.commit()
            return JSONResponse({"reply": final, "title": conv.title, "model": active_model})

        coding_mode = is_coding_or_logic_query(user_text)
        if coding_mode:
            messages.append({"role": "system", "content": CODING_EXPERT_DIRECTIVE})

        # Pre-Flight RAG Grounding:
        # - Skip on personal profile queries
        # - On coding queries, only search if URLs are provided or user explicitly asks for docs/latest
        live_sources = []
        search_queries = [user_text]
        should_run_rag = not is_personal_or_account_query(user_text) and (
            not coding_mode or bool(extract_urls(user_text)) or any(w in lowered_text for w in ["docs", "documentation", "latest", "2026", "search"])
        )

        if should_run_rag:
            try:
                max_src = 6 if user_plan == "premium" else 4
                search_queries = generate_smart_search_queries(messages, user_text)
                urls_in_prompt = extract_urls(user_text)
                for u in urls_in_prompt[:2]:
                    page = await fetch_webpage_content(u)
                    if page:
                        messages.append({
                            "role": "system",
                            "content": f"USER PROVIDED LIVE WEBPAGE / TIKTOK ({page['url']}):\n{page['content']}"
                        })

                for q in search_queries:
                    found = await execute_python_rag(q, max_sources=max_src, include_news=not coding_mode)
                    if found:
                        for item in found:
                            if item["url"] not in [s["url"] for s in live_sources]:
                                live_sources.append(item)
                    if len(live_sources) >= max_src:
                        break

                if live_sources:
                    formatted_sources = []
                    for s in live_sources[:max_src]:
                        snippet_or_body = s.get("content") or s.get("snippet", "")
                        formatted_sources.append(
                            f"Source: [{s['title']}]({s['url']})\n"
                            f"Domain: {s.get('site', '')}\n"
                            f"Content Excerpt:\n{snippet_or_body[:1200]}\n---"
                        )

                    rag_block = "REAL-TIME VERIFIED WEB SOURCES & ARTICLE CONTENTS:\n" + "\n".join(formatted_sources)
                    messages.append({
                        "role": "system",
                        "content": (
                            f"{rag_block}\n\n"
                            "STRICT GROUNDING & ACCURACY DIRECTIVES:\n"
                            "1. Base your answer directly on the verified article and social excerpts above when answering external factual questions.\n"
                            "2. If asked about a creator, YouTuber, or meme trend, explain the actual online context. Do NOT invent fake slang definitions and do NOT lecture on reporting harassment.\n"
                            "3. Directly answer the user's question without stalling or repeatedly asking for more details.\n"
                            "4. Cite sources using clean markdown links [Source Title](URL).\n"
                            "5. Do NOT use markdown tables (no `| Detail | Information |`). Output clean paragraphs and bullet points."
                        )
                    })
            except Exception as rag_err:
                print(f"RAG pre-flight non-fatal warning: {rag_err}")

        try:
            completion = client.chat.completions.create(
                model=active_model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_out_tokens,
            )
        except Exception as first_err:
            err_text = str(first_err)
            print(f"Chat completion initial attempt notice: {err_text}")

            if "Tool choice is none" in err_text or "browser.run" in err_text or "tool_use_failed" in err_text:
                m = re.search(r'"query":\s*"([^"]+)"', err_text)
                retry_query = m.group(1) if m else search_queries[0]

                extra_sources = await execute_python_rag(retry_query)
                if extra_sources:
                    formatted_sources = [
                        f"Source: [{s['title']}]({s['url']})\nContent: {s.get('content', s.get('snippet', ''))[:1000]}"
                        for s in extra_sources[:4]
                    ]
                    rag_block = "REAL-TIME VERIFIED WEB SOURCES:\n" + "\n".join(formatted_sources)
                    messages.append({
                        "role": "system",
                        "content": f"ADDITIONAL SEARCH FINDINGS FOR '{retry_query}':\n{rag_block}"
                    })

                messages.append({
                    "role": "system",
                    "content": (
                        "CRITICAL: Do NOT output any tool calls, function calls, or syntax like 'browser.run'. "
                        "Answer the user's inquiry directly using conversational text and clean bullet points."
                    )
                })

                completion = client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=max_out_tokens,
                )
            else:
                raise first_err

        raw = completion.choices[0].message.content or ""

        # Premium 2-Pass Self-Critique Engine (For Premium users on coding & architecture tasks)
        if user_plan == "premium" and coding_mode and len(raw) > 120:
            try:
                review_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Reasoning: high\n"
                            f"{SYSTEM_PROMPT}\n"
                            "You are a Principal Software Architect reviewing a draft solution. "
                            "Check the draft for any bugs, missing imports, incomplete placeholders, or logic errors. "
                            "Output ONLY the final, perfected response and complete runnable code directly to the user."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"User Request:\n{user_text}\n\nDraft Solution to Verify & Perfect:\n{raw}"
                    }
                ]
                review_comp = client.chat.completions.create(
                    model=MODEL_PAID,
                    messages=review_messages,
                    temperature=0.2,
                    max_tokens=max_out_tokens,
                )
                reviewed_raw = (review_comp.choices[0].message.content or "").strip()
                if reviewed_raw and not is_refusal_message(reviewed_raw):
                    raw = reviewed_raw
            except Exception as rev_err:
                print(f"Premium 2-pass review notice: {rev_err}")

        # Automatic Single-Turn Refusal Loop Escape
        if is_refusal_message(raw) and not is_slur_or_restricted_name(user_text):
            try:
                now_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")
                reasoning_level = get_reasoning_effort_for_plan(user_plan)
                fresh_system = (
                    f"Reasoning: {reasoning_level}\n"
                    f"{SYSTEM_PROMPT}\n\n"
                    f"TEMPORAL REALITY: Today is {now_str}.\n"
                    f"USER CONTEXT:\n{user_context_block(user)}\n\n"
                    "DIRECTIVE: Answer the user's latest message directly and helpfully without refusing."
                )
                fresh_messages = [
                    {"role": "system", "content": fresh_system},
                    {"role": "user", "content": user_text},
                ]
                fresh_comp = client.chat.completions.create(
                    model=active_model,
                    messages=fresh_messages,
                    temperature=0.3,
                    max_tokens=max_out_tokens,
                )
                raw = fresh_comp.choices[0].message.content or raw
            except Exception as retry_err:
                print(f"Fresh single-turn retry notice: {retry_err}")

        # False-Positive Denial Override
        sources_text = " ".join([s.get("content", "") + " " + s.get("title", "") for s in live_sources]).lower()
        if ("spacex" in sources_text or "rocket launch" in sources_text) and any(
            denial in raw.lower() for denial in ["wasn't a spacex", "not a spacex", "was not a spacex", "water-spume"]
        ):
            try:
                recovery_prompt = list(messages)
                recovery_prompt.append({
                    "role": "system",
                    "content": (
                        "OVERRIDE: The verified news sources state that the event WAS a SpaceX rocket launch creating a twilight jellyfish cloud. "
                        "Do not deny the SpaceX launch and do not mention water-spume. Confirm the SpaceX Falcon 9 launch factually."
                    )
                })
                retry_completion = client.chat.completions.create(
                    model=active_model,
                    messages=recovery_prompt,
                    temperature=0.2,
                )
                raw = retry_completion.choices[0].message.content or raw
            except Exception as override_err:
                print(f"Denial override warning: {override_err}")

        final = sanitize_response(raw)

        db.add(Message(conversation_id=conv.id, role="assistant", content=final))
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()

        return JSONResponse({
            "reply": final,
            "title": conv.title,
            "model": active_model,
        })
    except Exception as e:
        print(f"Chat completion error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# UPLOAD (VISION-TO-CODE & SCREENSHOT BUG FIXER FOR PLUS & PREMIUM)
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    conversation_id: int = Form(...),
    caption: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_verified(user)
    user_plan = (user.plan or "free").lower()
    if not can_use_images(user_plan):
        return JSONResponse(
            {"error": "Image & screenshot uploads require Plus ($4/mo) or Premium ($8/mo). Upgrade in Plans."},
            status_code=403,
        )
    if not client:
        return JSONResponse({"error": "Server missing GROQ_API_KEY"}, status_code=500)

    try:
        conv = get_user_conversation(db, user, conversation_id)

        content = b""
        chunk_size = 1024 * 1024
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            content += chunk
            if len(content) > MAX_FILE_SIZE:
                return JSONResponse({"error": "Image too large (max 20 MB)"}, status_code=400)

        try:
            img = Image.open(io.BytesIO(content))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail((2048, 2048))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            content = buf.getvalue()
            mime = "image/jpeg"
        except Exception as e:
            return JSONResponse({"error": f"Failed to process image: {e}"}, status_code=400)

        b64 = base64.b64encode(content).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"

        vision_completion = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image thoroughly:\n"
                            "1. If it contains code, terminal/console error logs, stack traces, or IDE screenshots, "
                            "transcribe every line of visible code, exact error messages, and line numbers.\n"
                            "2. If it is a UI design or website screenshot, describe the layout, components, colors, and typography in detail.\n"
                            "3. Otherwise, describe all important objects, text, setting, and details accurately."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0.2,
            max_tokens=1000,
        )
        image_description = vision_completion.choices[0].message.content.strip()

        note = f"[User uploaded an image/screenshot]\nExtracted Image Details:\n{image_description}"
        if caption and caption.strip():
            note += f"\n\nUser Request: {caption.strip()}"

        msg_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
        if msg_count == 0 and conv.title == "New chat":
            conv.title = make_title_from_text(caption or "Image analysis")

        db.add(Message(conversation_id=conv.id, role="user", content=note))
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()

        # Plus runs at Reasoning: medium; Premium runs at Reasoning: high
        messages = build_messages_for_conversation(db, user, conv.id)
        messages.append({"role": "system", "content": CODING_EXPERT_DIRECTIVE})
        active_model = resolve_model_for_user(user_plan, model)
        max_out_tokens = get_max_tokens_for_plan(user_plan)

        try:
            completion = client.chat.completions.create(
                model=active_model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_out_tokens,
            )
        except Exception as first_err:
            err_text = str(first_err)
            if "Tool choice is none" in err_text or "browser.run" in err_text:
                messages.append({
                    "role": "system",
                    "content": "Do NOT emit tool calls or browser.run. Answer directly in conversational text."
                })
                completion = client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=max_out_tokens,
                )
            else:
                raise first_err

        raw = completion.choices[0].message.content or ""
        final = sanitize_response(raw)
        db.add(Message(conversation_id=conv.id, role="assistant", content=final))
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()
        return JSONResponse({
            "reply": final,
            "title": conv.title,
            "filename": file.filename,
            "size": len(content),
        })
    except Exception as e:
        print(f"Upload error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/reset")
async def reset(
    conversation_id: int = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv = get_user_conversation(db, user, conversation_id)
    db.query(Message).filter(Message.conversation_id == conv.id).delete()
    conv.title = "New chat"
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# CASH APP CHECKOUT & AUTO-VERIFICATION ROUTES
# ---------------------------------------------------------------------------
@app.post("/create-checkout")
async def create_checkout(
    plan: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_verified(user)
    plan = plan.strip().lower()
    if plan not in ("plus", "premium"):
        return JSONResponse({"error": "Invalid plan"}, status_code=400)
    if not CASHAPP_TAG:
        return JSONResponse({"error": "CASHAPP_TAG is not configured in environment variables"}, status_code=500)

    if not user.payment_code:
        user.payment_code = f"DAED-{user.id}-{secrets.token_hex(2).upper()}"
    user.pending_plan = plan
    db.commit()

    amount = 4 if plan == "plus" else 8
    clean_tag = CASHAPP_TAG.lstrip("$")
    encoded_note = urllib.parse.quote(user.payment_code)

    cashapp_url = f"https://cash.app/${clean_tag}/{amount}?note={encoded_note}"

    return JSONResponse({
        "cashtag": f"${clean_tag}",
        "amount": amount,
        "plan": plan,
        "code": user.payment_code,
        "cashapp_url": cashapp_url,
    })


@app.post("/verify-cashapp")
async def verify_cashapp(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_verified(user)
    db.refresh(user)

    if not user.payment_code:
        return JSONResponse({"verified": False, "error": "No pending upgrade found. Select a plan first."}, status_code=400)

    if not EMAIL_USER or not EMAIL_APP_PASSWORD:
        return JSONResponse({"verified": False, "error": "Server email verification credentials not configured yet."}, status_code=500)

    target_plan = user.pending_plan or "plus"

    matched_plan = await asyncio.to_thread(
        check_imap_for_cashapp_receipt,
        user.payment_code,
        target_plan,
    )

    if matched_plan in ("plus", "premium"):
        user.plan = matched_plan
        user.payment_code = None
        user.pending_plan = None
        db.commit()
        return JSONResponse({
            "verified": True,
            "plan": user.plan,
            "message": f"Payment verified! Your account has been upgraded to {user.plan.upper()}."
        })

    return JSONResponse({
        "verified": False,
        "message": "Waiting for Cash App payment receipt... Make sure your code is in the 'For' note."
    })
