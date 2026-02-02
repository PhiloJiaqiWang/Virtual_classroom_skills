# Virtual Classroom (Teacher Agent)

A lightweight FastAPI app that provides a teacher-style chat UI with configurable prompt methods, session history, and optional ASCII diagrams in responses.

## Features
- Chat UI with MathJax rendering.
- Multiple teaching methods (Direct Instruction, Socratic, etc.).
- Manage methods in a dedicated page (create/edit/delete/set default).
- Session history saved to a local file; viewable in a History page.
- Optional ASCII diagrams in responses using fenced ` ```ascii` blocks.

## Project Structure
- `server.py` — FastAPI backend and OpenAI integration.
- `static/index.html` — main chat UI.
- `static/methods.html` — methods management UI.
- `static/history.html` — history viewer UI.
- `methods.json` — saved prompt methods (auto-created).
- `history.jsonl` — saved chat sessions (auto-created).
- `.env.example` — environment variable template.

## Requirements
- Python 3.9+ recommended
- `fastapi`, `uvicorn`, `openai`, `python-dotenv`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup
1) Create a `.env` file:
```bash
cp .env.example .env
```
2) Add your OpenAI API key in `.env`:
```
OPENAI_API_KEY=your_key_here
```

## Run
```bash
uvicorn server:app --reload
```

Open:
- Chat UI: `http://127.0.0.1:8000/`
- Methods page: `http://127.0.0.1:8000/methods-page`
- History page: `http://127.0.0.1:8000/history-page`

## Notes
- New methods are stored in `methods.json`.
- Sessions are saved to `history.jsonl` on reset, method change, or when closing the page.
- If you want to reset history, delete `history.jsonl`.
