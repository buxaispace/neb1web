import os
import re
import secrets
import base64
import io
import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
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

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

# ---------------------------------------------------------------------------
# Config & Environment Variables
# ---------------------------------------------------------------------------
API_KEY = os.getenv("GROQ_API_KEY", "").strip()
API_KEY_2 = os.getenv("GROQ_API_KEY_2", "").strip()
API_KEY_3 = os.getenv("GROQ_API_KEY_3", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

DATABASE_URL = os.getenv("DATABASE_URL", "")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
APP_URL = os.getenv("APP_URL", "").rstrip("/")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "onboarding@resend.dev")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

CASHAPP_TAG = os.getenv("CASHAPP_TAG", "").strip()
EMAIL_USER = os.getenv("EMAIL_USER", "").strip()
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "").replace(" ", "").strip()
EMAIL_IMAP_HOST = os.getenv("EMAIL_IMAP_HOST", "imap.gmail.com").strip()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

groq_clients: List[Groq] = []
for k in [API_KEY, API_KEY_2, API_KEY_3]:
    if k:
        try:
            groq_clients.append(Groq(api_key=k))
        except Exception:
            pass

client = groq_clients[0] if groq_clients else None

MODEL_NAME = "Daedalus"
FULL_IDENTITY = "Daedalus"
MODEL_FREE = "openai/gpt-oss-20b"
MODEL_PAID = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"
MAX_FILE_SIZE = 20 * 1024 * 1024
HISTORY_LIMIT = 24

SYSTEM_PROMPT = (
    f"You are {FULL_IDENTITY}, an advanced AI system engineered by AstroVectAI for software engineering, "
    "computer science, physics, orbital mechanics, aerospace, mathematics, and deep technical research.\n"
    "CORE OPERATIONAL DIRECTIVES:\n"
    "1. NO UNPROMPTED SELF-INTRODUCTIONS (CRITICAL):\n"
    "   - Start your response by directly answering the user's question or solving their problem.\n"
    "   - NEVER start normal responses with 'Hello! I am Daedalus, developed by AstroVectAI, not ChatGPT' or any similar disclaimer.\n"
    f"   - ONLY state your identity ('I am {FULL_IDENTITY}, developed by AstroVectAI.') if the user explicitly asks who you are or what model you are.\n"
    "2. TONE & PRECISION:\n"
    "   - Communicate like a Principal Engineer and Chief Scientist: clear, rigorous, well-structured, and direct.\n"
    "   - Answer ALL parts of a multi-part prompt thoroughly.\n"
    "3. GREETING & NAME RULES:\n"
    "   - NEVER greet users using website domain names, URLs, or email handles.\n"
    "   - Do NOT shoehorn saved project codenames or user memories into unrelated greetings unless relevant to the question.\n"
    "   - If the user asks what their username or display name is, tell them their name from USER CONTEXT.\n"
    "4. CLEAN PRESENTATION & FORMATTING RULES (STRICT):\n"
    "   - NEVER use markdown tables (`| Col1 | Col2 |`). Present structured data using clean bullet points (`• **Label:** Value`).\n"
    "   - NEVER use raw LaTeX delimiters like `\\( ... \\)`, `\\[ ... \\]`, or `\\text{...}`. Write equations using clean Unicode math symbols (e.g., `R_E = 6,378.137 km`, `v = √(μ / r)`, `Δv`, `×`, `≈`, `²`, `³`) so they read cleanly on screen.\n"
    "   - Whenever you write code, wrap it inside triple-backtick markdown code fences with the language name (```html, ```python, etc.).\n"
    "5. LIVE WEB & NEWS VERIFICATION (NEVER CLAIM YOU CANNOT BROWSE):\n"
    "   - Your server automatically searches the live web and injects real-time sources sorted newest-first into your context.\n"
    "   - NEVER say 'I don\\'t have the ability to browse the web' or 'My knowledge cutoff is September 2024'.\n"
    "   - ALWAYS prioritize the most recent dates and flight numbers (e.g., higher flight numbers and newest dates in the index supersede older flights).\n"
    "   - ONLY cite exact URLs provided in the `MULTI-SOURCE LIVE WEB & NEWS FACT-CHECK INDEX`. NEVER invent fake URLs or generic homepage links like `[SpaceX Official Website](https://www.spacex.com)`.\n"
    "6. FIRST-PRINCIPLES STEM, SCIENCE, MATH & ENGINEERING RIGOR:\n"
    "   - For Physics, Space Science, Orbital Mechanics, Math, and Engineering: State the governing equations, keep SI units consistent, verify sign conventions (`+` vs `-`), show step-by-step substitutions, and sanity-check the final numbers.\n"
    "7. NO-TOOL / NO-FILESYSTEM DIRECTIVE (CRITICAL):\n"
    "   - NEVER output `<tool_call>`, `<function=...>`, `list_files`, `read_file`, or `browser.run`.\n"
    "8. CONTEXTUAL COMMON SENSE & ANTI-FALSE-POSITIVE SAFETY POLICY (CRITICAL):\n"
    "   - Interpret words like 'cutting', 'hook', 'barb', 'line', 'rigging', 'pin', 'wire', 'load', 'kill process', or 'execute' in their practical DIY, fishing, crafting, mechanical, or software context.\n"
    "   - NEVER output suicide helplines (988) or self-harm warnings for benign prompts about fishing gear, modifying hooks/barbs, attaching lines to clothing/props, hardware projects, or everyday physical tasks.\n"
    "   - Reserve crisis helpline responses STRICTLY for explicit expressions of suicidal intent or deliberate self-injury.\n"
    "   - Evaluate every new user message independently. Never repeat a refusal or helpline message from an earlier turn."
)

STEM_AND_ENGINEERING_DIRECTIVE = (
    "UNIVERSAL SOFTWARE ARCHITECTURE, ENGINEERING & STEM ENGINE ACTIVE:\n"
    "1. INVARIANT, SIGN & MATHEMATICAL VERIFICATION (ALL DOMAINS):\n"
    "   - Verify sign conventions (`+` vs `-`), units, and state invariants (e.g., inflows/income add to balance, outflows/expenses/subscriptions subtract from balance; forces and vectors point in the right direction).\n"
    "   - Format equations cleanly using readable Unicode symbols (`v = √(μ / r)`, `T = 2π √(r³ / μ)`, `× 10¹⁴ m³/s²`) instead of raw LaTeX `\\( ... \\)`.\n"
    "2. IF BUILDING A FRONTEND / WEB APP (HTML/CSS/JS):\n"
    "   - Design a complete, modern interactive product using Tailwind CSS CDN (`<script src=\"https://cdn.tailwindcss.com\"></script>`), Lucide Icons (`<script src=\"https://unpkg.com/lucide@latest\"></script>`), and Google Fonts (`Inter` & `JetBrains Mono`).\n"
    "   - Use a responsive multi-column Bento Grid layout (`lg:grid-cols-12 gap-6`), pre-populate 3–4 realistic demo items if `localStorage` is empty on first boot, tuck settings inside a modal drawer, and include floating toast notifications.\n"
    "3. IF WRITING BACKEND, ALGORITHMS, SIMULATIONS, OR SOLVING SCIENCE/MATH:\n"
    "   - Prioritize optimal complexity, numerical stability (e.g. RK4 for physics/orbits), step-by-step derivations, and zero placeholders (`// TODO` is forbidden). Wrap code in ```language fences."
)

# ---------------------------------------------------------------------------
# Multi-Key Groq + OpenRouter Failover + Automatic 413 TPM Compaction
# ---------------------------------------------------------------------------
def compact_messages_for_tpm(messages: List[dict]) -> List[dict]:
    if len(messages) <= 2:
        return messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    chat_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]

    primary_sys = system_msgs[0].copy() if system_msgs else {"role": "system", "content": SYSTEM_PROMPT}
    if len(system_msgs) > 1:
        extra_sys = "\n\n".join(m["content"][:1100] for m in system_msgs[1:])
        primary_sys["content"] = primary_sys["content"] + "\n\n" + extra_sys

    recent_chat = []
    for m in chat_msgs[-2:]:
        c = m["content"]
        if len(c) > 10500:
            c = c[:10500] + "\n... [truncated for token limit]"
        recent_chat.append({"role": m["role"], "content": c})
    return [primary_sys] + recent_chat


def call_external_openrouter(model: str, messages: List[dict], temperature: float) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        return None
    candidates = [f"{model}:free" if not model.endswith(":free") else model, model]
    for or_model in candidates:
        try:
            with httpx.Client(timeout=75.0) as http:
                r = http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": APP_URL or "https://astrovect.online",
                        "X-Title": "Daedalus AI",
                    },
                    json={"model": or_model, "messages": messages, "temperature": temperature},
                )
                if r.status_code == 200:
                    content = r.json().get("choices", [{}])[0].get("message", {}).get("content")
                    if content:
                        return content
        except Exception as or_err:
            print(f"OpenRouter failover notice ({or_model}): {or_err}")
    return None


def call_llm_with_failover(
    model: str,
    messages: List[dict],
    temperature: float = 0.3,
    max_tokens: int = 6000,
) -> str:
    last_err = None
    hit_tpm_or_rate_limit = False

    for idx, g_client in enumerate(groq_clients):
        try:
            comp = g_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return comp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if any(k in err_str for k in ["413", "429", "rate_limit", "too large", "tokens per minute", "tpm", "quota"]):
                hit_tpm_or_rate_limit = True
                continue
            else:
                raise e

    if OPENROUTER_API_KEY:
        or_reply = call_external_openrouter(model, messages, temperature)
        if or_reply:
            return or_reply

    if hit_tpm_or_rate_limit and groq_clients:
        compacted_msgs = compact_messages_for_tpm(messages)
        for g_client in groq_clients:
            try:
                comp = g_client.chat.completions.create(
                    model=model,
                    messages=compacted_msgs,
                    temperature=temperature,
                    max_tokens=min(max_tokens, 4096),
                )
                return comp.choices[0].message.content or ""
            except Exception as retry_e:
                last_err = retry_e
                continue

        alt_model = MODEL_PAID if model == MODEL_FREE else MODEL_FREE
        for g_client in groq_clients:
            try:
                comp = g_client.chat.completions.create(
                    model=alt_model,
                    messages=compacted_msgs,
                    temperature=temperature,
                    max_tokens=4096,
                )
                return comp.choices[0].message.content or ""
            except Exception:
                continue

    if last_err:
        raise last_err
    raise RuntimeError("No LLM API keys configured.")


# ---------------------------------------------------------------------------
# Database & Models (With Per-Chat Memory + User-Only Info)
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
    chat_memory = Column(Text, nullable=True)
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
    if not engine:
        return
    try:
        Base.metadata.create_all(bind=engine)
        migrations = [
            ("users", "memory", "TEXT"),
            ("users", "payment_code", "VARCHAR(100)"),
            ("users", "pending_plan", "VARCHAR(50)"),
            ("conversations", "chat_memory", "TEXT"),
        ]
        with engine.connect() as conn:
            for tbl, col_name, col_type in migrations:
                try:
                    conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                    conn.commit()
                except Exception:
                    try:
                        conn.rollback()
                        conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN {col_name} {col_type};"))
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


# ---------------------------------------------------------------------------
# TIER CAPABILITIES & TOKEN BUDGETS
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
    user_plan = (plan or "free").lower()
    if user_plan == "premium":
        return "high"
    if user_plan == "plus":
        return "medium"
    return "low"


def get_max_tokens_for_plan(plan: str, is_coding: bool = False) -> int:
    user_plan = (plan or "free").lower()
    if user_plan == "premium":
        return 16384 if is_coding else 8192
    if user_plan == "plus":
        return 12288 if is_coding else 6144
    return 4096


def can_use_images(plan: str) -> bool:
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


def strip_thinking_and_tool_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else text.strip()


def ensure_code_fences(text: str) -> str:
    if not text:
        return ""
    stripped = strip_thinking_and_tool_tags(text)

    fence_count = len(re.findall(r"```", stripped))
    if fence_count % 2 == 1:
        return stripped + "\n```"
    if fence_count >= 2:
        return stripped

    if re.match(r"^(?:<!DOCTYPE\s+html|<html\b|<head\b|<body\b|<style\b|<div\b)", stripped, flags=re.IGNORECASE):
        return f"```html\n{stripped}\n```"

    lines = stripped.splitlines()
    if len(lines) >= 4:
        first_line = lines[0].strip()
        if re.match(r"^(?:import\s+\w+|from\s+[\w\.]+\s+import\s+|def\s+\w+\s*\(|async\s+def\s+\w+\s*\(|class\s+\w+)", first_line):
            return f"```python\n{stripped}\n```"
        if re.match(r"^(?:const\s+\w+|let\s+\w+|function\s+\w+|export\s+default|import\s+.*from\s+['\"])", first_line):
            return f"```javascript\n{stripped}\n```"

    return stripped


def clean_markdown_tables_outside_code(text_segment: str) -> str:
    if not text_segment:
        return ""
    try:
        lines = text_segment.splitlines()
        new_lines = []
        table_header_words = {
            "detail", "details", "property", "field", "attribute", "key", "topic",
            "category", "header", "agency", "event", "date", "source", "mission",
            "parameter", "value", "unit", "units", "symbol", "description", "metric",
            "name", "status", "quantity", "formula", "step", "item"
        }
        for line in lines:
            stripped = line.strip()
            if re.match(r"^[•\-\*]\s*(?:Agency\s*—\s*Event|Parameter\s*—\s*Value|Detail\s*—\s*Information|Property\s*—\s*Value)", stripped, flags=re.IGNORECASE):
                continue

            if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
                raw_parts = stripped.split("|")[1:-1]
                parts = [p.strip() for p in raw_parts]
                if not parts or not any(parts):
                    continue
                if all(re.match(r"^:?-+:?$", p) for p in parts if p):
                    continue
                cleaned_cols = [re.sub(r"\*+", "", p).strip().lower() for p in parts if p]
                if cleaned_cols and sum(1 for c in cleaned_cols if c in table_header_words) >= max(1, len(cleaned_cols) // 2):
                    continue

                if len(parts) == 2:
                    key, val = parts[0], parts[1]
                    val = re.sub(r"<br\s*/?>", "\n  ", val)
                    key_clean = re.sub(r"^\*\*|\*\*$", "", key).strip()
                    new_lines.append(f"• **{key_clean}:** {val}")
                elif len(parts) > 2:
                    lead = re.sub(r"^\*\*|\*\*$", "", parts[0]).strip()
                    rest = " · ".join(p for p in parts[1:] if p)
                    new_lines.append(f"• **{lead}:** {rest}")
                elif len(parts) == 1 and parts[0]:
                    new_lines.append(f"• {parts[0]}")
            else:
                new_lines.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(new_lines))
    except Exception:
        return text_segment


def clean_latex_outside_code(text_segment: str) -> str:
    """Converts raw LaTeX math delimiters into clean readable scientific notation."""
    if not text_segment:
        return ""
    s = text_segment
    s = re.sub(r"\\\[\s*([\s\S]*?)\s*\\\]", r"\1", s)
    s = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"\1", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1 / \2)", s)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", s)
    replacements = [
        (r"\\,", ""),
        (r"\\;", " "),
        (r"\\!", ""),
        (r"\\quad", "  "),
        (r"\\times", "×"),
        (r"\\cdot", "·"),
        (r"\\approx", "≈"),
        (r"\\neq", "≠"),
        (r"\\leq", "≤"),
        (r"\\geq", "≥"),
        (r"\\pm", "±"),
        (r"\\pi", "π"),
        (r"\\mu", "μ"),
        (r"\\Delta", "Δ"),
        (r"\\delta", "δ"),
        (r"\\theta", "θ"),
        (r"\\omega", "ω"),
        (r"\\infty", "∞"),
        (r"\\circ", "°"),
    ]
    for pat, rep in replacements:
        s = re.sub(pat, rep, s)
    return s


def clean_markdown_and_math(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(```[\s\S]*?```)", text)
    out = []
    for part in parts:
        if part.startswith("```") and part.endswith("```"):
            out.append(part)
