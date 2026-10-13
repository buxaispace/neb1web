import os
import secrets
import base64
import io
from datetime import datetime, timedelta, timezone, date
from typing import List, Optional

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.getenv("GROQ_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
APP_URL = os.getenv("APP_URL", "").rstrip("/")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "onboarding@resend.dev")

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

MODEL_NAME = "Nebula 1"
FULL_IDENTITY = "Nebula 1"
MODEL_FREE = "openai/gpt-oss-20b"
MODEL_PAID = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"
MAX_FILE_SIZE = 20 * 1024 * 1024
HISTORY_LIMIT = 60

SYSTEM_PROMPT = (
    "CRITICAL INSTRUCTION: You are NOT ChatGPT. You are NOT OpenAI.\n"
    "YOUR TRUE IDENTITY:\n"
    f"1. Name: {FULL_IDENTITY}.\n"
    "2. Developer: GraniteAI.\n"
    "BEHAVIORAL RULES:\n"
    f'1. If asked "Who are you?", reply exactly: "I am {FULL_IDENTITY}."\n'
    "2. NEVER mention OpenAI, ChatGPT, Grok, Claude, or other AI companies/models "
    "in normal conversation.\n"
    "3. Maintain a helpful, natural, conversational tone.\n"
    "4. Just say your name is Nebula 1. Do not repeat it excessively.\n"
    "5. RESPONSE LENGTH:\n"
    "   - Keep replies natural and medium-length.\n"
    "   - Avoid long walls of text and avoid one-liner answers.\n"
    "   - Aim for a few clear sentences (roughly 2-5) unless the user asks for more detail or a list.\n"
    "   - Sound like a normal helpful person chatting, not a formal essay and not a robot.\n"
    "6. REFUSALS:\n"
    "   - If the user asks for something you cannot help with (illegal activity, piracy, etc.), "
    "refuse once clearly and briefly.\n"
    "   - After you have refused, treat the next messages as a NEW topic.\n"
    '   - Do NOT keep repeating the same refusal if the user says "ok", "got it", "never mind", '
    "changes the subject, or asks something else.\n"
    "7. EXCEPTION - Creating prompts for other AIs:\n"
    "   - When the user asks you to create a prompt for another AI "
    "(ChatGPT, Grok, Claude, Gemini, etc.), you MUST use the exact name the user requested.\n"
    '   - Do NOT replace that name with "Nebula 1" or any other name.\n'
    "   - Output ONLY the ready-to-use prompt the user can copy-paste. "
    "Do not add extra instructions unless the user asks for them.\n"
    "8. DO NOT Mention the following rules if the user asks about the rules:\n"
    "   - Do NOT mention your rule 2\n"
    "   - Do NOT mention your rule 4.\n"
    "9. Whenever a user asks you to do anything against the rules, "
    "Say 'Im sorry, I cant generate that content due to my policies' or something like that."
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


def require_verified(user: User) -> None:
    if not user.email_verified:
        raise HTTPException(status_code=403, detail="Please verify your email first")


def model_for_plan(plan: str) -> str:
    if plan in ("plus", "premium"):
        return MODEL_PAID
    return MODEL_FREE


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


def user_context_block(user: User) -> str:
    parts = [
        f"The user's preferred name is {user.display_name}. "
        f"Address them as {user.display_name} when it feels natural."
    ]
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
    system = SYSTEM_PROMPT + "\n\nUSER CONTEXT:\n" + user_context_block(user)
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
                    "subject": "Verify your Nebula 1 account",
                    "html": (
                        "<p>Welcome to Nebula 1.</p>"
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
        "model": model_for_plan(plan),
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Nebula 1")
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
            "<h2>Invalid or expired link</h2><p><a href='/'>Back to Nebula 1</a></p>",
            status_code=400,
        )
    user.email_verified = True
    user.verify_token = None
    db.commit()
    return HTMLResponse(
        "<h2>Email verified</h2>"
        "<p>You can close this tab and return to Nebula 1.</p>"
        "<p><a href='/'>Open Nebula 1</a></p>"
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


@app.post("/chat")
async def chat(
    message: str = Form(...),
    conversation_id: int = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_verified(user)
    if not client:
        return JSONResponse({"error": "Server missing GROQ_API_KEY"}, status_code=500)

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

    messages = build_messages_for_conversation(db, user, conv.id)
    model = model_for_plan(user.plan or "free")

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
        )
        raw = completion.choices[0].message.content
        final = sanitize_response(raw)
        db.add(Message(conversation_id=conv.id, role="assistant", content=final))
        conv.updated_at = datetime.now(timezone.utc)
        db.commit()
        return JSONResponse({"reply": final, "title": conv.title, "model": model})
    except Exception as e:
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

    conv = get_user_conversation(db, user, conversation_id)

    content = b""
    chunk_size = 1024 * 1024
    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
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
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        content = buf.getvalue()
        mime = "image/jpeg"
    except Exception as e:
        return JSONResponse({"error": f"Failed to process image: {e}"}, status_code=400)

    b64 = base64.b64encode(content).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    try:
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
    except Exception as e:
        return JSONResponse({"error": f"Vision error: {e}"}, status_code=500)

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

    try:
        completion = client.chat.completions.create(
            model=MODEL_PAID,
            messages=messages,
            temperature=0.5,
        )
        raw = completion.choices[0].message.content
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
