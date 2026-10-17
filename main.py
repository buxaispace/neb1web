import os
import re
import secrets
import base64
import io
import urllib.parse
from html import unescape
from datetime import datetime, timedelta, timezone, date
from typing import List, Optional, Dict, Any

import httpx
import stripe
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
    create_engine, Column, Integer, String, Text, DateTime, Date, Boolean, ForeignKey,
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
    "2. IF ADDRESSED AS CHATGPT (e.g. 'hey chatgpt', 'hello chatgpt', 'are you chatgpt'):\n"
    "   - State your identity clearly and politely: 'I am Daedalus, developed by AstroVectAI, not ChatGPT. How can I help you today?'\n"
    "   - Do NOT enthusiastically play along or say generic greetings like 'Hey <domain>! I am Daedalus—what can I do for you today?'.\n"
    "3. NEVER pretend to be ChatGPT, OpenAI, Grok, Claude, or other AI models.\n"
    "4. Maintain a helpful, natural, conversational, and factually accurate tone.\n"
    "5. GREETING & NAME RULES:\n"
    "   - NEVER greet users using website domain names, URLs, or email handles (e.g., NEVER say 'Hey buxai.space!').\n"
    "   - Do NOT awkwardly force a user's name into every greeting or response.\n"
    "6. FORMATTING RULES (STRICT):\n"
    "   - NEVER use markdown tables (do NOT use `| Detail | Information |` or any table with `|` pipes).\n"
    "   - Present all information using clean natural prose or standard bullet points (`•`).\n"
    "7. TEMPORAL ACCURACY & SEARCH GROUNDING:\n"
    "   - You have active, real-time internet search capability.\n"
    "   - Today's date is in October 2026. Do NOT state that your knowledge cuts off in 2023 or 2024.\n"
    "   - When answering questions about current events, facts, or specifications, "
    "rely strictly on real-time search findings.\n"
    "   - When citing sources, use real, verified Markdown links [Source Title](URL). "
    "NEVER hallucinate or guess fake URLs.\n"
    "8. REFUSALS:\n"
    "   - If the user asks for illegal or harmful content, refuse once clearly and briefly.\n"
    "   - Treat subsequent messages as a new topic.\n"
    "9. EXCEPTION - Creating prompts for other AIs:\n"
    "   - When the user asks you to create a prompt for another AI "
    "(ChatGPT, Grok, Claude, Gemini, etc.), you MUST use the exact name the user requested.\n"
    '   - Do NOT replace that name with "Daedalus". Output ONLY the ready-to-use prompt.\n'
    "10. Whenever a user asks you to do anything against the rules, "
    "Say 'I am sorry, I cannot generate that content due to my policies.'"
)

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
    display_name = Column(String(100), nullable=False, default="Friend")
    birthday = Column(Date, nullable=True)
    plan = Column(String(50), default="free")
    email_verified = Column(Boolean, default=False)
    verify_token = Column(String(100), nullable=True)
    stripe_customer_id = Column(String(100), nullable=True)
    memory = Column(Text, nullable=True)
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
):
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


def require_verified(user: User) -> None:
    if not user.email_verified:
        raise HTTPException(status_code=403, detail="Please verify your email first")


def resolve_model_for_user(plan: str, requested_model: Optional[str] = None) -> str:
    """Enforces tier model permissions. Free users are locked to gpt-oss-20b."""
    user_plan = (plan or "free").lower()
    if user_plan not in ("plus", "premium"):
        return MODEL_FREE

    # Plus and Premium can choose 20b or 120b
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
            "but do not awkwardly force their name into greetings or every response."
        )

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
    }


# ---------------------------------------------------------------------------
# RAG ENGINE
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


def clean_html(html: str, max_chars: int = 2500) -> str:
    if not html:
        return ""
    text = re.sub(r'<(script|style|svg|noscript|header|footer|nav|aside)[^>]*>.*?</\1>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = unescape(text)
    return re.sub(r'\s+', ' ', text).strip()[:max_chars]


async def fetch_webpage(url: str, max_chars: int = 2500) -> Optional[Dict[str, str]]:
    try:
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True, headers=BROWSER_HEADERS) as http:
            res = await http.get(url)
            if res.status_code == 200:
                html = res.text
                title_match = re.search(r'<title[^>]*>(.*?)</title>', html, flags=re.IGNORECASE | re.DOTALL)
                title = unescape(title_match.group(1)).strip() if title_match else url
                body = clean_html(html, max_chars)
                if body:
                    return {
                        "title": title[:100],
                        "url": url,
                        "snippet": body[:300] + "...",
                        "content": body,
                        "site": urllib.parse.urlparse(url).netloc,
                    }
    except Exception as e:
        print(f"Error reading {url}: {e}")
    return None


async def execute_python_rag(query: str) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []

    if TAVILY_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=6.0) as http:
                r = await http.post(
                    "https://api.tavily.com/search",
                    json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 4},
                )
                if r.status_code == 200:
                    for item in r.json().get("results", []):
                        sources.append({
                            "title": item.get("title", "Source"),
                            "url": item.get("url", ""),
                            "snippet": item.get("content", ""),
                            "site": urllib.parse.urlparse(item.get("url", "")).netloc,
                        })
                    if sources:
                        return sources
        except Exception as e:
            print("Tavily search warning:", e)

    if SERPER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=6.0) as http:
                r = await http.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={"q": query, "num": 4},
                )
                if r.status_code == 200:
                    for item in r.json().get("organic", [])[:4]:
                        sources.append({
                            "title": item.get("title", "Google Result"),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "site": urllib.parse.urlparse(item.get("link", "")).netloc,
                        })
                    if sources:
                        return sources
        except Exception as e:
            print("Serper search warning:", e)

    if HAS_DDGS:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=4):
                    href = r.get("href", "")
                    if href.startswith("http"):
                        sources.append({
                            "title": r.get("title", "Web Result"),
                            "url": href,
                            "snippet": r.get("body", ""),
                            "site": urllib.parse.urlparse(href).netloc,
                        })
                if sources:
                    return sources
        except Exception as e:
            print("DDGS search warning:", e)

    try:
        async with httpx.AsyncClient(timeout=6.0, headers=BROWSER_HEADERS) as http:
            q_enc = urllib.parse.quote_plus(query)
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q_enc}&format=json&utf8="
            r = await http.get(wiki_url)
            if r.status_code == 200:
                data = r.json()
                for item in data.get("query", {}).get("search", [])[:3]:
                    page_title = item.get("title", "")
                    clean_snippet = clean_html(item.get("snippet", ""), 300)
                    sources.append({
                        "title": page_title,
                        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}",
                        "snippet": clean_snippet,
                        "site": "en.wikipedia.org",
                    })
                if sources:
                    return sources
    except Exception as e:
        print("Wikipedia search warning:", e)

    return sources


# ---------------------------------------------------------------------------
# App & Template Routes
# ---------------------------------------------------------------------------
app = FastAPI(title="Daedalus")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
# CHAT ENDPOINT (WITH MODEL SELECTION ENFORCEMENT)
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
        conv = get_user_conversation(db, user, conversation_id)
        user_text = message.strip()
        if not user_text:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        msg_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
        if msg_count == 0 and conv.title == "New chat":
            conv.title = make_title_from_text(user_text)

        # Save user message
        db.add(Message(conversation_id=conv.id, role="user", content=user_text))
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()

        # Build conversation history
        messages = build_messages_for_conversation(db, user, conv.id)

        # Resolve model according to tier rules (Free locked to 20b)
        active_model = resolve_model_for_user(user.plan or "free", model)

        # Check for direct identity greeting test ("hey chatgpt")
        lowered_text = user_text.lower().strip()
        if lowered_text in ["hey chatgpt", "hello chatgpt", "hi chatgpt", "chatgpt"]:
            final = "I am Daedalus, developed by AstroVectAI, not ChatGPT. How can I help you today?"
            db.add(Message(conversation_id=conv.id, role="assistant", content=final))
            conv.updated_at = datetime.now(timezone.utc)
            db.commit()
            return JSONResponse({"reply": final, "title": conv.title, "model": active_model})

        # Pre-Flight RAG Grounding
        try:
            urls_in_prompt = extract_urls(user_text)
            for u in urls_in_prompt[:2]:
                page = await fetch_webpage(u)
                if page:
                    messages.append({
                        "role": "system",
                        "content": f"USER PROVIDED LIVE WEBPAGE ({page['url']}):\n{page['content']}"
                    })

            # Fetch real-time web context
            live_sources = await execute_python_rag(user_text)
            if live_sources:
                rag_block = "REAL-TIME VERIFIED WEB SOURCES:\n" + "\n".join([
                    f"- [{s['title']}]({s['url']}): {s.get('snippet', '')}" for s in live_sources[:4]
                ])
                messages.append({
                    "role": "system",
                    "content": (
                        f"{rag_block}\n\n"
                        "GROUNDING DIRECTIVE:\n"
                        "- Use the above live web findings as ground truth.\n"
                        "- Cite claims with markdown links [Source Title](URL).\n"
                        "- Do NOT use markdown tables (no `| Detail | Information |`)."
                    )
                })
        except Exception as rag_err:
            print(f"RAG pre-flight non-fatal warning: {rag_err}")

        # Execute Completion
        completion = client.chat.completions.create(
            model=active_model,
            messages=messages,
            temperature=0.3,
        )

        raw = completion.choices[0].message.content or ""
        final = sanitize_response(raw)

        # Save assistant message
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


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    conversation_id: int = Form(...),
    caption: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_verified(user)
    if not can_use_images(user.plan or "free"):
        return JSONResponse(
            {"error": "Image upload requires Premium ($8/mo). Upgrade in Plans."},
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
                            "Describe this image in clear detail. "
                            "Include important objects, text, colors, setting, "
                            "and any notable details. Be factual and thorough but concise."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0.2,
            max_tokens=500,
        )
        image_description = vision_completion.choices[0].message.content.strip()

        note = f"[User uploaded an image]\nImage description: {image_description}"
        if caption and caption.strip():
            note += f"\nUser said: {caption.strip()}"

        msg_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
        if msg_count == 0 and conv.title == "New chat":
            conv.title = make_title_from_text(caption or "Image chat")

        db.add(Message(conversation_id=conv.id, role="user", content=note))
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()

        messages = build_messages_for_conversation(db, user, conv.id)

        completion = client.chat.completions.create(
            model=MODEL_PAID,
            messages=messages,
            temperature=0.3,
        )
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
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"error": "Payments not configured yet"}, status_code=500)

    price_id = STRIPE_PRICE_PLUS if plan == "plus" else STRIPE_PRICE_PREMIUM
    if not price_id:
        return JSONResponse({"error": f"Stripe price for {plan} not set"}, status_code=500)
    if not APP_URL:
        return JSONResponse({"error": "APP_URL not set"}, status_code=500)

    if not user.stripe_customer_id:
        customer = stripe.Customer.create(
            email=user.email,
            metadata={"user_id": str(user.id)},
        )
        user.stripe_customer_id = customer.id
        db.commit()

    session = stripe.checkout.Session.create(
        customer=user.stripe_customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{APP_URL}/?checkout=success",
        cancel_url=f"{APP_URL}/?checkout=cancel",
        metadata={"user_id": str(user.id), "plan": plan},
        subscription_data={"metadata": {"user_id": str(user.id), "plan": plan}},
        payment_method_types=["card"],
    )
    return JSONResponse({"url": session.url})


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    if not STRIPE_WEBHOOK_SECRET:
        return JSONResponse({"error": "Webhook not configured"}, status_code=500)
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        meta = session.get("metadata") or {}
        user_id = meta.get("user_id")
        plan = meta.get("plan")
        if user_id and plan in ("plus", "premium"):
            user = db.query(User).filter(User.id == int(user_id)).first()
            if user:
                user.plan = plan
                if session.get("customer"):
                    user.stripe_customer_id = session["customer"]
                db.commit()

    elif event["type"] in ("customer.subscription.deleted", "customer.subscription.updated"):
        sub = event["data"]["object"]
        meta = sub.get("metadata") or {}
        user_id = meta.get("user_id")
        status = sub.get("status")
        if user_id:
            user = db.query(User).filter(User.id == int(user_id)).first()
            if user:
                if status in ("canceled", "unpaid", "incomplete_expired"):
                    user.plan = "free"
                elif status == "active":
                    plan = meta.get("plan")
                    if plan in ("plus", "premium"):
                        user.plan = plan
                db.commit()

    return JSONResponse({"received": True})
