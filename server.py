import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional
import json
from pathlib import Path
from datetime import datetime, timezone
import uuid

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "OPENAI_API_KEY not found. Put it in .env"

client = OpenAI(api_key=api_key)

app = FastAPI()

# Serve the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
METHOD_PROMPTS = {
    "direct_instruction": (
        "You are a patient teacher using Direct Instruction. "
        "Explain step by step. Start with a clear definition, then 1-2 key points, "
        "then a concrete example, then a quick summary. "
        "Ask at most one clarifying question if needed. "
        "If a diagram would help, include a short ASCII diagram in a fenced block labeled ```ascii```."
    ),
    "scaffolding": (
        "You are a teacher using Scaffolding. "
        "Break the concept into small steps. After each step, ask a brief check-for-understanding question. "
        "Adapt based on the user's response. Use simple language and examples. "
        "If a diagram would help, include a short ASCII diagram in a fenced block labeled ```ascii```."
    ),
    "socratic": (
        "You are a teacher using the Socratic method. "
        "Do not lecture first. Ask guiding questions to help the user derive the idea. "
        "Keep questions short, one at a time. Provide a mini-explanation only after the user attempts. "
        "If a diagram would help, include a short ASCII diagram in a fenced block labeled ```ascii```."
    ),
    "worked_example": (
        "You are a teacher using Worked Examples. "
        "Give one fully worked example with numbered steps, then give a similar practice question. "
        "If the user asks, show the solution gradually. "
        "If a diagram would help, include a short ASCII diagram in a fenced block labeled ```ascii```."
    ),
    "analogy_first": (
        "You are a teacher using Analogy-First instruction. "
        "Start with an everyday analogy, map analogy parts to the concept, "
        "then give a formal definition and a small example. "
        "If a diagram would help, include a short ASCII diagram in a fenced block labeled ```ascii```."
    ),
}
DEFAULT_METHOD = "direct_instruction"
SKILLS_PATH = Path("skills.json")
SKILLS_INTRO_PATH = Path("skillsIntro.json")
METHODS_PATH = Path("methods.json")
HISTORY_PATH = Path("history.jsonl")

# In-memory chat history (single-user, simple demo)
# For multiple users, we'd store per-session histories.
history: list[dict] = []
current_method: str = DEFAULT_METHOD
session_id: str = ""
session_started_at: str = ""
session_method: str = DEFAULT_METHOD

def _default_methods_payload():
    return {
        "default": DEFAULT_METHOD,
        "methods": [
            {"id": k, "label": k.replace("_", " ").title(), "prompt": v}
            for k, v in METHOD_PROMPTS.items()
        ],
    }

_INTRO_TEMPLATES = {
    "direct_instruction": "Step-by-step explanation with a definition, key points, an example, and a brief summary.",
    "scaffolding": "Breaks ideas into small steps with quick check-for-understanding questions.",
    "socratic": "Uses guiding questions to help the learner discover the idea before a short explanation.",
    "worked_example": "Shows a fully worked example, then gives a similar practice problem.",
    "analogy_first": "Starts with an everyday analogy, maps it to the concept, then formalizes it.",
    "philo": "Explains complex ideas in very simple, child-friendly language.",
}

def _build_intro_for(method: dict):
    mid = method.get("id", "").strip()
    label = method.get("label", "").strip()
    prompt = method.get("prompt", "").strip()
    if mid in _INTRO_TEMPLATES:
        return _INTRO_TEMPLATES[mid]
    first_sentence = prompt.split(".")[0].strip()
    if first_sentence:
        return first_sentence + "."
    return f"Briefly explains {label or mid}."

def _default_skills_intro(methods: list[dict]):
    return {
        "methods": [
            {"id": m["id"], "label": m.get("label", m["id"]), "intro": _build_intro_for(m)}
            for m in methods
        ]
    }

def _load_skills_intro(methods: list[dict]):
    if SKILLS_INTRO_PATH.exists():
        try:
            data = json.loads(SKILLS_INTRO_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "methods" in data:
                existing = {m.get("id"): m for m in data.get("methods", [])}
                merged = []
                for m in methods:
                    mid = m["id"]
                    if mid in existing and existing[mid].get("intro"):
                        merged.append(
                            {
                                "id": mid,
                                "label": existing[mid].get("label", m.get("label", mid)),
                                "intro": existing[mid]["intro"],
                            }
                        )
                    else:
                        merged.append(
                            {"id": mid, "label": m.get("label", mid), "intro": _build_intro_for(m)}
                        )
                data = {"methods": merged}
                SKILLS_INTRO_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return data
        except json.JSONDecodeError:
            pass
    data = _default_skills_intro(methods)
    SKILLS_INTRO_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data

def _load_methods_payload():
    if SKILLS_PATH.exists():
        try:
            data = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "methods" in data:
                return data
        except json.JSONDecodeError:
            pass
    if METHODS_PATH.exists():
        try:
            data = json.loads(METHODS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "methods" in data:
                SKILLS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return data
        except json.JSONDecodeError:
            pass
    data = _default_methods_payload()
    SKILLS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data

def _apply_methods_payload(data: dict):
    global METHOD_PROMPTS, DEFAULT_METHOD, current_method
    methods = data.get("methods", [])
    METHOD_PROMPTS = {m["id"]: m["prompt"] for m in methods}
    DEFAULT_METHOD = data.get("default") or (methods[0]["id"] if methods else "")
    if DEFAULT_METHOD not in METHOD_PROMPTS and methods:
        DEFAULT_METHOD = methods[0]["id"]
    if current_method not in METHOD_PROMPTS:
        current_method = DEFAULT_METHOD

METHODS_DATA = _load_methods_payload()
_apply_methods_payload(METHODS_DATA)
SKILLS_INTRO = _load_skills_intro(METHODS_DATA.get("methods", []))

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _start_session(method_id: str):
    global session_id, session_started_at, session_method
    session_id = str(uuid.uuid4())
    session_started_at = _now_iso()
    session_method = method_id

def _choose_method_id(user_msg: str):
    methods = SKILLS_INTRO.get("methods", [])
    ids = [m.get("id") for m in methods if m.get("id")]
    if not ids:
        return DEFAULT_METHOD, []

    options = "\n".join([f"- {m['id']}: {m.get('intro','')}" for m in methods])
    router_messages = [
        {
            "role": "system",
            "content": (
                "You choose the best teaching method for a user's message. "
                "Return only the method id from the list. No extra words."
            ),
        },
        {"role": "user", "content": f"Message: {user_msg}\n\nMethods:\n{options}"},
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=router_messages,
            temperature=0,
            max_tokens=20,
        )
        choice = (resp.choices[0].message.content or "").strip().split()[0]
        return (choice if choice in ids else DEFAULT_METHOD), router_messages
    except Exception:
        return DEFAULT_METHOD, router_messages

def _append_session(reason: str):
    payload = {
        "id": session_id,
        "started_at": session_started_at,
        "ended_at": _now_iso(),
        "method": session_method,
        "reason": reason,
        "messages": history,
    }
    HISTORY_PATH.write_text("", encoding="utf-8") if not HISTORY_PATH.exists() else None
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _load_history():
    if not HISTORY_PATH.exists():
        return []
    items = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    items.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return items

_start_session(current_method)

SYSTEM_PROMPT = (
    "You are a patient teacher. Explain step by step, use simple language, "
    "and give a concrete example. If the user is unclear, ask one short clarifying question. "
    "If a diagram would help, include a short ASCII diagram in a fenced block labeled ```ascii```."
)

class ChatRequest(BaseModel):
    message: str = ""
    reset: bool = False
    method: Optional[str] = None

class MethodItem(BaseModel):
    id: str
    label: str
    prompt: str

class MethodsUpdate(BaseModel):
    default: Optional[str] = None
    methods: list[MethodItem]

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/methods-page")
def methods_page():
    return FileResponse("static/methods.html")

@app.get("/history-page")
def history_page():
    return FileResponse("static/history.html")

@app.post("/chat")
def chat(req: ChatRequest):
    global history, current_method, session_method

    # Reset requested
    if req.reset:
        _append_session("reset")
        history = []
        _start_session(current_method)
        return {"reply": "Memory cleared. Let's start fresh.", "method": current_method}

    # Choose method: auto-select unless a method is explicitly provided
    routing_messages = []
    if req.method:
        chosen = req.method.strip() or current_method
        if chosen not in METHOD_PROMPTS:
            chosen = DEFAULT_METHOD
        if chosen != current_method:
            _append_session("method_change")
            current_method = chosen
            history = []
            _start_session(current_method)
    else:
        chosen, routing_messages = _choose_method_id(req.message or "")
        current_method = chosen
        session_method = chosen

    user_msg = (req.message or "").strip()
    if not user_msg:
        return {"reply": "Type something and I'll help 🙂", "method": current_method}

    if not METHOD_PROMPTS:
        return {"reply": "No prompt methods configured. Please add one in the Methods page.", "method": current_method}

    system_prompt = METHOD_PROMPTS[current_method]
    methods_list = METHODS_DATA.get("methods", [])
    label_map = {m["id"]: m.get("label", m["id"]) for m in methods_list}
    method_label = label_map.get(current_method, current_method)

    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": user_msg}
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )

    reply = resp.choices[0].message.content
    reply = f"[{method_label}] {reply}"

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "method": current_method,
        "debug": {
            "chosen_method": current_method,
            "routing_messages": routing_messages,
            "system_prompt": system_prompt,
            "messages": messages,
        },
    }

@app.get("/methods")
def methods():
    methods_list = METHODS_DATA.get("methods", [])
    return {
        "default": DEFAULT_METHOD,
        "methods": methods_list,
    }

@app.get("/history")
def history_api():
    return {"history": _load_history()}

@app.post("/session-end")
def session_end():
    if history:
        _append_session("exit")
        history.clear()
        _start_session(current_method)
    return {"ok": True}

@app.put("/methods")
def update_methods(payload: MethodsUpdate):
    global METHODS_DATA, SKILLS_INTRO
    methods = payload.methods
    if not methods:
        raise HTTPException(status_code=400, detail="At least one method is required.")

    ids = [m.id.strip() for m in methods]
    if any(not mid for mid in ids):
        raise HTTPException(status_code=400, detail="Method id cannot be empty.")
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="Method ids must be unique.")

    methods_clean = [
        {"id": m.id.strip(), "label": m.label.strip() or m.id.strip(), "prompt": m.prompt.strip()}
        for m in methods
    ]
    if any(not m["prompt"] for m in methods_clean):
        raise HTTPException(status_code=400, detail="Method prompt cannot be empty.")

    default_id = (payload.default or ids[0]).strip()
    if default_id not in [m["id"] for m in methods_clean]:
        default_id = methods_clean[0]["id"]

    METHODS_DATA = {"default": default_id, "methods": methods_clean}
    SKILLS_PATH.write_text(json.dumps(METHODS_DATA, indent=2), encoding="utf-8")
    _apply_methods_payload(METHODS_DATA)
    SKILLS_INTRO = _load_skills_intro(METHODS_DATA.get("methods", []))

    return {"ok": True, "default": DEFAULT_METHOD, "methods": METHODS_DATA["methods"]}
