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
    "   - Your server automatically searches the live web and injects real-time sources into your context.\n"
    "   - NEVER say 'I don\\'t have the ability to browse the web' or 'My knowledge cutoff is September 2024'. Today is October 2026.\n"
    "   - ONLY cite exact URLs provided in the `MULTI-SOURCE LIVE WEB & NEWS FACT-CHECK INDEX`. NEVER invent fake URLs or generic homepage links like `[SpaceX Official Website](https://www.spacex.com)`. If a specific detail isn't in the live sources, state what the provided sources actually report.\n"
    "6. FIRST-PRINCIPLES STEM, SCIENCE, MATH & ENGINEERING RIGOR:\n"
    "   - For Physics, Space Science, Orbital Mechanics, Math, and Engineering: State the governing equations, keep SI units consistent, verify sign conventions (`+` vs `-`), show step-by-step substitutions, and sanity-check the final numbers.\n"
    "7. NO-TOOL / NO-FILESYSTEM DIRECTIVE (CRITICAL):\n"
    "   - NEVER output `<tool_call>`, `<function=...>`, `list_files`, `read_file`, or `browser.run`.\n"
    "8. REFUSAL & FRESH-TURN POLICY:\n"
    "   - Evaluate every new user message independently. Never repeat a refusal from an earlier turn unless the new message itself directly asks for severe illegal harm."
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
            # Also strip pre-converted "• Agency — Event — Date — Source" header lines
            if re.match(r"^[•\-\*]\s*(?:Agency\s*—\s*Event|Parameter\s*—\s*Value|Detail\s*—\s*Information|Property\s*—\s*Value)", stripped, flags=re.IGNORECASE):
                continue

            if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
                raw_parts = stripped.split("|")[1:-1]
                parts = [p.strip() for p in raw_parts]
                if not parts or not any(parts):
                    continue
                # Skip markdown separator rows like |---|---|
                if all(re.match(r"^:?-+:?$", p) for p in parts if p):
                    continue
                # Skip table header rows where columns are generic labels
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
    # Strip display and inline math delimiters \( ... \) and \[ ... \]
    s = re.sub(r"\\\[\s*([\s\S]*?)\s*\\\]", r"\1", s)
    s = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"\1", s)
    # Clean \text{...} and \mathrm{...}
    s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    # Convert \frac{a}{b} -> (a / b)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1 / \2)", s)
    # Convert \sqrt{x} -> √(x)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", s)
    # Common symbols & spacing
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
        else:
            seg = clean_markdown_tables_outside_code(part)
            seg = clean_latex_outside_code(seg)
            out.append(seg)
    return "".join(out).strip()


def sanitize_response(text: str, allow_identity_intro: bool = False) -> str:
    forbidden = [
        "large language model trained by",
        "As an AI",
        "language model trained by",
        "I am a large language model",
        "I'm a large language model",
    ]
    cleaned = strip_thinking_and_tool_tags(text)
    for phrase in forbidden:
        if phrase in cleaned:
            cleaned = cleaned.replace(phrase, MODEL_NAME)

    # Strip unprompted "Hello! I'm Daedalus, developed by AstroVectAI, not ChatGPT..." openers
    if not allow_identity_intro:
        cleaned = re.sub(
            r"^\s*(?:(?:Hello|Hi|Hey|Greetings)[!,\.]?\s*)?(?:I am|I'm|I’m)\s+Daedalus(?:,\s*developed by AstroVectAI)?(?:,\s*not ChatGPT)?[\.\!]?\s*(?:How can I help you[^\n\?\!]*[\?\!]?\s*)?(?:---\s*)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

    cleaned = ensure_code_fences(cleaned)
    cleaned = clean_markdown_and_math(cleaned)

    cleaned = re.sub(
        r"^(Hey|Hello|Hi)\s+[a-zA-Z0-9_\-\.]+\.(space|com|org|net|io|ai|app|dev|co|online)[!,\.]*\s*",
        r"\1! ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def is_direct_chatgpt_greeting(user_text: str) -> bool:
    """Returns True ONLY if the user directly addresses the AI as ChatGPT."""
    lower = user_text.lower().strip()
    if lower in {"chatgpt", "hey chatgpt", "hello chatgpt", "hi chatgpt", "yo chatgpt", "ok chatgpt"}:
        return True
    if re.match(r"^(?:hey|hi|hello|yo|dear|ok)\s+chatgpt\b", lower):
        return True
    return False


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
    if any(suffix in name.lower() for suffix in [".space", ".com", ".org", ".net", ".io", ".ai", ".app", ".dev", ".co", ".online"]):
        return None
    if "@" in name or "/" in name:
        return None
    if len(name) < 1 or len(name) > 100:
        return None
    return name


def user_and_chat_context_block(user: User, conv: Conversation) -> str:
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
                f"The user's display name / username is '{clean_name}' (only mention it if they ask what their name/username is)."
            )

    if user.birthday:
        parts.append(
            f"Their birthday is {user.birthday.isoformat()} (do not bring it up unless asked)."
        )
    if user.memory and user.memory.strip():
        parts.append(f"Background notes about this user (reference only when relevant): {user.memory.strip()}")

    if conv.chat_memory and conv.chat_memory.strip():
        parts.append(
            f"Session notes for this chat (reference ONLY when directly relevant; do not force into greetings): {conv.chat_memory.strip()}"
        )
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
        "don't have the ability to browse",
        "don’t have the ability to browse",
        "cannot browse the web",
        "can't browse the web",
        "can’t browse the web",
        "last update in september 2024",
        "knowledge cutoff",
    ]
    return len(lower) < 320 and any(p in lower for p in refusal_phrases)


def build_messages_for_conversation(db: Session, user: User, conv: Conversation) -> List[dict]:
    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
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
        f"USER & SESSION CONTEXT:\n{user_and_chat_context_block(user, conv)}"
    )
    history = [{"role": "system", "content": system}]

    cleaned_rows = []
    i = 0
    while i < len(rows):
        m = rows[i]
        if m.role == "user" and (i + 1) < len(rows) and rows[i + 1].role == "assistant":
            if is_refusal_message(rows[i + 1].content) or "<tool_call>" in rows[i + 1].content:
                i += 2
                continue
        if m.role == "assistant" and (is_refusal_message(m.content) or "<tool_call>" in m.content):
            i += 1
            continue
        cleaned_rows.append(m)
        i += 1

    total_rows = len(cleaned_rows)
    for idx, m in enumerate(cleaned_rows):
        content = m.content
        is_recent_turn = idx >= (total_rows - 2)
        if not is_recent_turn and len(content) > 2500:
            content = content[:2500] + "\n... [earlier code omitted to save context window]"
        history.append({"role": m.role, "content": content})
    return history


def is_stem_or_coding_query(text: str) -> bool:
    lower = text.lower()
    if "```" in text:
        return True
    stem_keywords = [
        "code", "script", "python", "javascript", "typescript", "html", "css",
        "react", "nextjs", "fastapi", "flask", "django", "sql", "postgres",
        "luau", "roblox", "unity", "c#", "c++", "java", "rust", "golang",
        "function", "variable", "bug", "error", "traceback", "exception",
        "syntax", "compile", "algorithm", "regex", "json", "api endpoint",
        "write a program", "make a website", "build a game", "debug", "fix this",
        "single-file", "app with", "pomodoro", "tracker", "dashboard", "calculator",
        "calculate", "equation", "formula", "derivative", "integral", "matrix",
        "physics", "orbit", "orbital", "rocket", "delta-v", "gravity", "relativity",
        "quantum", "thermodynamics", "astrophysics", "velocity", "trajectory",
        "engineering", "circuit", "theorem", "proof", "solve"
    ]
    return any(k in lower for k in stem_keywords)


def is_followup_code_edit(messages: List[dict], user_text: str) -> bool:
    for m in messages[:-1]:
        if m.get("role") == "assistant" and ("```" in m.get("content", "") or "<!DOCTYPE html>" in m.get("content", "")):
            return True
    return False


def generate_stem_architecture_blueprint(user_text: str) -> Optional[str]:
    try:
        spec_prompt = [
            {
                "role": "system",
                "content": (
                    "Reasoning: low\n"
                    "You are a Chief Scientist, Principal Systems Engineer & Staff UI Architect.\n"
                    "Analyze the user's request and output a concise, high-density Technical & Mathematical Blueprint (under 200 words) tailored to its exact domain:\n"
                    "- IF FRONTEND/WEB APP: Specify a multi-column responsive layout, interactive visualizations (SVG/Canvas), pre-populated realistic test data when storage is empty, a modal for settings, toast notifications, and strict state/math invariants (e.g. inflows add, outflows subtract).\n"
                    "- IF BACKEND / SYSTEMS / GAME CODE: Specify data structures, time/space complexity, edge-case guards, and modular functions.\n"
                    "- IF MATH / PHYSICS / SPACE / ENGINEERING: Specify the governing physical laws or equations (e.g., Vis-Viva, Tsiolkovsky, Newton/Kepler, calculus identities), SI unit conversions, sign/vector conventions, and a final order-of-magnitude sanity check.\n"
                    "Output ONLY the blueprint bullet points—no code yet."
                )
            },
            {"role": "user", "content": user_text}
        ]
        blueprint = call_llm_with_failover(
            model=MODEL_PAID,
            messages=spec_prompt,
            temperature=0.2,
            max_tokens=1500,
        )
        return strip_thinking_and_tool_tags(blueprint)
    except Exception as e:
        print(f"STEM blueprint notice: {e}")
        return None


def update_user_and_chat_memories(db: Session, user: User, conv: Conversation, user_text: str):
    if not (client or OPENROUTER_API_KEY):
        return
    lower = user_text.lower()

    personal_triggers = ["i am a ", "i'm a ", "i live in", "my age is", "about me:", "my background is"]
    chat_triggers = [
        "remember that", "remember our", "for this project", "in this chat",
        "keep in mind", "our goal is", "codename is", "mission is"
    ]

    try:
        if any(t in lower for t in personal_triggers):
            current_umem = (user.memory or "").strip()
            note = call_llm_with_failover(
                model=MODEL_FREE,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Reasoning: low\n"
                            "Extract 1 brief personal fact about the user themselves (e.g., their role, background, or personal trait). "
                            "Do NOT include chat-specific topics or tasks. Output ONLY the short note (max 15 words)."
                        )
                    },
                    {"role": "user", "content": user_text}
                ],
                temperature=0.1,
                max_tokens=150,
            ).strip()
            if note and note not in current_umem:
                user.memory = ((current_umem + " | " + note).strip(" |"))[:3000]
                db.commit()

        if any(t in lower for t in chat_triggers):
            current_cmem = (conv.chat_memory or "").strip()
            cnote = call_llm_with_failover(
                model=MODEL_FREE,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Reasoning: low\n"
                            "Extract 1 brief note about what is being worked on or remembered in this specific chat session (such as codenames, variables, or project goals). "
                            "Output ONLY the short note (max 20 words)."
                        )
                    },
                    {"role": "user", "content": user_text}
                ],
                temperature=0.1,
                max_tokens=150,
            ).strip()
            if cnote and cnote not in current_cmem:
                conv.chat_memory = ((current_cmem + " | " + cnote).strip(" |"))[:3000]
                db.commit()
    except Exception as e:
        print(f"Memory update notice: {e}")


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
                    if not is_from_cashapp or code_lower not in combined_lower:
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
# PARALLEL MULTI-SOURCE RAG & FACT-CHECKING ENGINE (NEWS + REDDIT + WIKIPEDIA + WEB)
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


def clean_html(html: str, max_chars: int = 2000) -> str:
    if not html:
        return ""
    text = re.sub(r'<(script|style|svg|noscript|header|footer|nav|aside|form)[^>]*>.*?</\1>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = unescape(text)
    return re.sub(r'\s+', ' ', text).strip()[:max_chars]


async def fetch_tiktok_video_data(url: str) -> Optional[Dict[str, str]]:
    try:
        async with httpx.AsyncClient(timeout=4.5, follow_redirects=True, headers=BROWSER_HEADERS) as http:
            oembed_url = f"https://www.tiktok.com/oembed?url={urllib.parse.quote(url)}"
            res = await http.get(oembed_url)
            if res.status_code == 200:
                data = res.json()
                title = data.get("title", "")
                author = data.get("author_name", "")
                handle = data.get("author_unique_id", "")
                content = f"TikTok Video by @{handle} ({author}): {title}"
                return {
                    "title": f"TikTok: @{handle} - {title[:60]}",
                    "url": url,
                    "snippet": content[:320],
                    "content": content,
                    "site": "tiktok.com"
                }
    except Exception:
        pass
    return None


async def fetch_webpage_content(url: str, max_chars: int = 1200) -> Optional[Dict[str, str]]:
    if "tiktok.com" in url:
        tiktok_data = await fetch_tiktok_video_data(url)
        if tiktok_data:
            return tiktok_data

    try:
        async with httpx.AsyncClient(timeout=4.5, follow_redirects=True, headers=BROWSER_HEADERS) as http:
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
                if body and len(body) > 80:
                    return {
                        "title": title[:100],
                        "url": url,
                        "snippet": body[:320],
                        "content": body,
                        "site": urllib.parse.urlparse(url).netloc,
                    }
    except Exception:
        pass
    return None


def is_personal_or_account_query(text: str) -> bool:
    lower = text.lower().strip()
    personal_patterns = [
        "my username", "my name", "call me", "who am i", "what about now",
        "updated now", "my birthday", "my account", "my plan", "who are you",
        "your name", "what is your name", "our mission codename", "what is the codename"
    ]
    if any(p in lower for p in personal_patterns):
        return True
    if len(lower.split()) <= 2 and not extract_urls(text):
        return True
    return False


def generate_smart_search_queries(messages: List[dict], user_text: str) -> List[str]:
    """
    Extracts targeted search queries from the full prompt without cutting off at 90 characters,
    stripping out memory instructions ('In this chat, remember...') and math sub-prompts.
    """
    queries = []

    cleaned = re.sub(
        r'in this chat[^\.!\?\n]*[\.!\?\n]*',
        ' ',
        user_text,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r'\b(first|second|third|fact-check|fact check|cite your sources|and cite|calculate the exact|showing the|step-by-step|why not|what happened to|tell me about|who is|can you|the one with|back in)\b',
        ' ',
        cleaned,
        flags=re.IGNORECASE
    )

    sentences = [s.strip() for s in re.split(r'[\.!\?\n]+', cleaned) if len(s.strip()) > 4]
    for s in sentences:
        s_lower = s.lower()
        if any(m_word in s_lower for m_word in ["orbital velocity", "km/s", "altitude of", "vis-viva"]):
            continue
        s_clean = re.sub(r'\s+', ' ', s).strip()
        if len(s_clean) > 4:
            queries.append(s_clean[:100])

    cap_words = re.findall(r'\b[A-Z][A-Za-z0-9\-]{2,}\b', cleaned)
    stop_caps = {"In", "First", "Second", "Third", "The", "This", "That", "What", "Who", "Where", "When", "Why", "How"}
    entities = [w for w in cap_words if w not in stop_caps]
    if entities:
        entity_query = " ".join(dict.fromkeys(entities))
        if "news" in user_text.lower() or "latest" in user_text.lower() or "week" in user_text.lower():
            entity_query += " latest news 2026"
        if entity_query not in queries:
            queries.insert(0, entity_query[:100])

    if not queries:
        user_turns = [m["content"] for m in messages if m.get("role") == "user"]
        combined = " ".join(user_turns[-2:])
        queries.append(re.sub(r'\s+', ' ', combined).strip()[:100])

    lower_u = user_text.lower()
    if any(k in lower_u for k in ["why is everyone calling", "meme", "gay", "drama", "tiktok", "youtuber", "creator"]):
        queries.append(f"{queries[0][:60]} tiktok reddit")

    return queries[:3]


async def query_reddit_search(query: str) -> List[Dict[str, str]]:
    results = []
    try:
        reddit_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        q_enc = urllib.parse.quote_plus(query)
        url = f"https://www.reddit.com/search.json?q={q_enc}&sort=relevance&limit=3"
        async with httpx.AsyncClient(timeout=4.5, follow_redirects=True, headers=reddit_headers) as http:
            r = await http.get(url)
            if r.status_code == 200:
                for child in r.json().get("data", {}).get("children", [])[:3]:
                    d = child.get("data", {})
                    title = unescape(d.get("title", ""))
                    selftext = unescape(clean_html(d.get("selftext", ""), 320))
                    sub = d.get("subreddit_name_prefixed", "r/Reddit")
                    permalink = d.get("permalink", "")
                    post_url = f"https://www.reddit.com{permalink}" if permalink else d.get("url", "")
                    body_text = f"[{sub}] {title} — {selftext}"
                    results.append({
                        "title": f"Reddit ({sub}): {title[:75]}",
                        "url": post_url,
                        "snippet": body_text[:320],
                        "site": "reddit.com"
                    })
    except Exception:
        pass
    return results


async def query_google_news_rss(query: str) -> List[Dict[str, str]]:
    results = []
    try:
        async with httpx.AsyncClient(timeout=4.5, follow_redirects=True, headers=BROWSER_HEADERS) as http:
            q_enc = urllib.parse.quote_plus(query)
            rss_url = f"https://news.google.com/rss/search?q={q_enc}&hl=en-US&gl=US&ceid=US:en"
            r = await http.get(rss_url)
            if r.status_code == 200:
                root = ET.fromstring(r.text)
                for item in root.findall(".//item")[:4]:
                    title = unescape(re.sub(r'<[^>]+>', '', item.findtext("title") or "")).strip()
                    link = (item.findtext("link") or "").strip()
                    desc = unescape(re.sub(r'<[^>]+>', '', item.findtext("description") or "")).strip()
                    pub_date = (item.findtext("pubDate") or "")[:16]
                    source = item.findtext("source") or "News"
                    if title and link:
                        results.append({
                            "title": f"{title} ({pub_date})" if pub_date else title,
                            "url": link,
                            "snippet": desc[:300],
                            "site": source
                        })
    except Exception:
        pass
    return results


async def query_wikipedia_science(query: str) -> List[Dict[str, str]]:
    results = []
    try:
        async with httpx.AsyncClient(timeout=4.5, headers=BROWSER_HEADERS) as http:
            q_enc = urllib.parse.quote_plus(query)
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q_enc}&format=json&utf8=&srlimit=3"
            r = await http.get(wiki_url)
            if r.status_code == 200:
                for item in r.json().get("query", {}).get("search", [])[:3]:
                    page_title = item.get("title", "")
                    clean_snip = clean_html(item.get("snippet", ""), 320)
                    if page_title:
                        results.append({
                            "title": f"Wikipedia: {page_title}",
                            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}",
                            "snippet": clean_snip,
                            "site": "en.wikipedia.org",
                        })
    except Exception:
        pass
    return results


async def query_ddg_lite(query: str) -> List[Dict[str, str]]:
    results = []
    try:
        async with httpx.AsyncClient(timeout=4.5, follow_redirects=True, headers=BROWSER_HEADERS) as http:
            res = await http.post("https://lite.duckduckgo.com/lite/", data={"q": query})
            if res.status_code == 200:
                html_text = res.text
                links = re.findall(r'<a[^>]+class=[\'"]result-link[\'"][^>]+href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>', html_text, re.IGNORECASE | re.DOTALL)
                snippets = re.findall(r'<td[^>]+class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>', html_text, re.IGNORECASE | re.DOTALL)
                for i in range(min(len(links), len(snippets), 3)):
                    raw_url, raw_title = links[i]
                    if "uddg=" in raw_url:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                        url = parsed.get("uddg", [raw_url])[0]
                    else:
                        url = raw_url
                    clean_title = unescape(re.sub(r'<[^>]+>', '', raw_title).strip())
                    clean_snippet = unescape(re.sub(r'<[^>]+>', '', snippets[i]).strip())
                    if url.startswith("http") and clean_title:
                        results.append({
                            "title": clean_title,
                            "url": url,
                            "snippet": clean_snippet[:300],
                            "site": urllib.parse.urlparse(url).netloc
                        })
    except Exception:
        pass
    return results


async def execute_python_rag(query: str, max_sources: int = 7, include_news: bool = True) -> List[Dict[str, str]]:
    tasks = [
        query_reddit_search(query),
        query_wikipedia_science(query),
        query_ddg_lite(query),
    ]
    if include_news:
        tasks.insert(0, query_google_news_rss(query))

    batches = await asyncio.gather(*tasks, return_exceptions=True)
    sources: List[Dict[str, str]] = []
    for b in batches:
        if isinstance(b, list):
            sources.extend(b)

    if TAVILY_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=4.5) as http:
                r = await http.post(
                    "https://api.tavily.com/search",
                    json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 4},
                )
                if r.status_code == 200:
                    for item in r.json().get("results", []):
                        sources.append({
                            "title": item.get("title", "Source"),
                            "url": item.get("url", ""),
                            "snippet": item.get("content", "")[:320],
                            "site": urllib.parse.urlparse(item.get("url", "")).netloc,
                        })
        except Exception:
            pass

    unique_sources = []
    seen = set()
    for s in sources:
        u = s.get("url", "")
        if u and u not in seen:
            seen.add(u)
            unique_sources.append(s)

    top_to_crawl = unique_sources[:2]
    if top_to_crawl:
        crawled_pages = await asyncio.gather(
            *(fetch_webpage_content(s["url"]) for s in top_to_crawl),
            return_exceptions=True
        )
        for idx, crawled in enumerate(crawled_pages):
            if isinstance(crawled, dict) and crawled.get("content"):
                unique_sources[idx]["content"] = crawled["content"]
                unique_sources[idx]["url"] = crawled["url"]

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

        conv = Conversation(user_id=user.id, title="New chat", chat_memory=None)
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
            conv = Conversation(user_id=user.id, title="New chat", chat_memory=None)
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
    conv = Conversation(user_id=user.id, title="New chat", chat_memory=None)
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
        remaining = Conversation(user_id=user.id, title="New chat", chat_memory=None)
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
# CHAT ENDPOINT (MULTI-SOURCE FACT CHECKING + PER-CHAT MEMORY + STEM ENGINE)
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
    if not client and not OPENROUTER_API_KEY:
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

        update_user_and_chat_memories(db, user, conv, user_text)

        user_plan = (user.plan or "free").lower()
        active_model = resolve_model_for_user(user_plan, model)
        stem_or_coding_mode = is_stem_or_coding_query(user_text)
        max_out_tokens = get_max_tokens_for_plan(user_plan, is_coding=stem_or_coding_mode)

        messages = build_messages_for_conversation(db, user, conv)
        lowered_text = user_text.lower().strip()

        # Only reply with the "not ChatGPT" line if the user literally greeted it as ChatGPT
        if is_direct_chatgpt_greeting(user_text):
            if len(lowered_text.split()) <= 3:
                final = "I am Daedalus, developed by AstroVectAI, not ChatGPT. What are we building or solving today?"
                db.add(Message(conversation_id=conv.id, role="assistant", content=final))
                conv.updated_at = datetime.now(timezone.utc)
                db.commit()
                return JSONResponse({"reply": final, "title": conv.title, "model": active_model})
            else:
                messages.append({
                    "role": "system",
                    "content": "The user addressed you as ChatGPT. Briefly note in one short clause that you are Daedalus (by AstroVectAI), then immediately answer their full technical prompt."
                })

        if stem_or_coding_mode:
            blueprint_section = ""
            if user_plan in ("plus", "premium") and not is_followup_code_edit(messages, user_text):
                ux_blueprint = generate_stem_architecture_blueprint(user_text)
                if ux_blueprint:
                    blueprint_section = (
                        f"\n\nTECHNICAL & MATHEMATICAL ARCHITECTURE BLUEPRINT:\n{ux_blueprint}\n"
                        "Follow this blueprint with 100% precision."
                    )
            messages.append({
                "role": "system",
                "content": STEM_AND_ENGINEERING_DIRECTIVE + blueprint_section
            })

        # Multi-Source Fact-Checking RAG
        live_sources = []
        search_queries = [user_text]
        should_run_rag = not is_personal_or_account_query(user_text) and (
            not stem_or_coding_mode
            or bool(extract_urls(user_text))
            or any(w in lowered_text for w in [
                "docs", "documentation", "latest", "2026", "search", "news",
                "mission", "nasa", "spacex", "data", "constant", "fact-check", "fact check", "this week"
            ])
        )

        if should_run_rag:
            try:
                max_src = 8 if user_plan == "premium" else 6
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
                    found = await execute_python_rag(q, max_sources=max_src, include_news=True)
                    if found:
                        for item in found:
                            if item["url"] not in [s["url"] for s in live_sources]:
                                live_sources.append(item)
                    if len(live_sources) >= max_src:
                        break

                if live_sources:
                    formatted_sources = []
                    for idx, s in enumerate(live_sources[:max_src], 1):
                        snippet_or_body = s.get("content") or s.get("snippet", "")
                        formatted_sources.append(
                            f"[{idx}] [{s['title']}]({s['url']}) ({s.get('site', 'Web')})\n"
                            f"Excerpt: {snippet_or_body[:450]}"
                        )

                    rag_block = "MULTI-SOURCE LIVE WEB & NEWS FACT-CHECK INDEX:\n" + "\n\n".join(formatted_sources)
                    messages.append({
                        "role": "system",
                        "content": (
                            f"{rag_block}\n\n"
                            "STRICT FACT-CHECKING & CITATION DIRECTIVE:\n"
                            "1. ONLY report news events, dates, and mission names that explicitly appear in the sources above.\n"
                            "2. Cite each verified item using the exact URL from the index above: `[Article/Source Title](URL)`. NEVER fabricate generic homepage links like `[SpaceX Official Website](https://www.spacex.com)`.\n"
                            "3. Format each news item as a clean bullet point (`• **Agency / Topic (Date):** Summary — [Source](URL)`). Do NOT use markdown tables.\n"
                            "4. Answer ALL parts of the user's prompt (both news/fact-checking AND any math/physics calculations requested)."
                        )
                    })
            except Exception as rag_err:
                print(f"RAG pre-flight non-fatal warning: {rag_err}")

        try:
            raw = call_llm_with_failover(
                model=active_model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_out_tokens,
            )
        except Exception as first_err:
            err_text = str(first_err)
            if "Tool choice is none" in err_text or "browser.run" in err_text or "tool_use_failed" in err_text:
                m = re.search(r'"query":\s*"([^"]+)"', err_text)
                retry_query = m.group(1) if m else search_queries[0]

                extra_sources = await execute_python_rag(retry_query, max_sources=6)
                if extra_sources:
                    formatted_sources = [
                        f"- [{s['title']}]({s['url']}): {s.get('content', s.get('snippet', ''))[:400]}"
                        for s in extra_sources[:6]
                    ]
                    messages.append({
                        "role": "system",
                        "content": f"VERIFIED SEARCH FINDINGS FOR '{retry_query}':\n" + "\n".join(formatted_sources)
                    })

                messages.append({
                    "role": "system",
                    "content": "CRITICAL: Do NOT output tool calls or 'browser.run'. Answer directly in clean technical text."
                })

                raw = call_llm_with_failover(
                    model=active_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=max_out_tokens,
                )
            else:
                raise first_err

        # Automatic Refusal / "Can't Browse" Escape
        if is_refusal_message(raw) and not is_slur_or_restricted_name(user_text):
            try:
                if not live_sources:
                    fallback_q = " ".join(generate_smart_search_queries(messages, user_text)[:1])
                    live_sources = await execute_python_rag(fallback_q, max_sources=5, include_news=True)

                src_text = ""
                if live_sources:
                    src_text = "\n\nLIVE WEB SOURCES:\n" + "\n".join(
                        f"- [{s['title']}]({s['url']}): {s.get('snippet', '')}" for s in live_sources[:5]
                    )

                now_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")
                reasoning_level = get_reasoning_effort_for_plan(user_plan)
                fresh_system = (
                    f"Reasoning: {reasoning_level}\n"
                    f"{SYSTEM_PROMPT}\n\n"
                    f"TEMPORAL REALITY: Today is {now_str}.\n"
                    f"USER & SESSION CONTEXT:\n{user_and_chat_context_block(user, conv)}"
                    f"{src_text}\n\n"
                    "CRITICAL OVERRIDE: Do NOT claim you cannot browse the web or that your knowledge stops in 2024. "
                    "Use the live web sources above (and your scientific/mathematical knowledge) to answer ALL parts of the user's prompt completely."
                )
                fresh_messages = [
                    {"role": "system", "content": fresh_system},
                    {"role": "user", "content": user_text},
                ]
                raw = call_llm_with_failover(
                    model=active_model,
                    messages=fresh_messages,
                    temperature=0.3,
                    max_tokens=max_out_tokens,
                ) or raw
            except Exception as retry_err:
                print(f"Fresh single-turn retry notice: {retry_err}")

        allow_intro = is_direct_chatgpt_greeting(user_text) or any(
            q in lowered_text for q in ["who are you", "what are you", "your name", "are you chatgpt"]
        )
        final = sanitize_response(raw, allow_identity_intro=allow_intro)

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
    if not client and not OPENROUTER_API_KEY:
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

        image_description = call_llm_with_failover(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image thoroughly:\n"
                            "1. If it contains code, terminal/console error logs, math equations, or diagrams, "
                            "transcribe every line of visible code, formulas, exact error messages, and values.\n"
                            "2. If it is a UI design or website screenshot, describe the layout, components, colors, and typography in detail.\n"
                            "3. Otherwise, describe all important objects, text, setting, and details accurately."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0.2,
            max_tokens=1500,
        ).strip()

        note = f"[User uploaded an image/screenshot]\nExtracted Image Details:\n{image_description}"
        if caption and caption.strip():
            note += f"\n\nUser Request: {caption.strip()}"

        msg_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
        if msg_count == 0 and conv.title == "New chat":
            conv.title = make_title_from_text(caption or "Image analysis")

        db.add(Message(conversation_id=conv.id, role="user", content=note))
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()

        messages = build_messages_for_conversation(db, user, conv)
        messages.append({"role": "system", "content": STEM_AND_ENGINEERING_DIRECTIVE})
        active_model = resolve_model_for_user(user_plan, model)
        max_out_tokens = get_max_tokens_for_plan(user_plan, is_coding=True)

        raw = call_llm_with_failover(
            model=active_model,
            messages=messages,
            temperature=0.3,
            max_tokens=max_out_tokens,
        )

        final = sanitize_response(raw, allow_identity_intro=False)
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
    conv.chat_memory = None
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
