"""Backend API for the AI twin chatbot.

The API exposes a single /chat endpoint that accepts the latest user message
plus conversation history, generates a reply using OpenAI, and records
unknown questions via Pushover push notifications.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Literal, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pypdf import PdfReader
from pydantic import BaseModel, Field

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI()


def read_linkedin_profile(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"LinkedIn PDF not found at {path}")
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def read_summary(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found at {path}")
    return path.read_text(encoding="utf-8")


BASE_DIR = Path(__file__).parent
summary_text = read_summary(BASE_DIR / "me" / "summary.txt")
linkedin_text = read_linkedin_profile(BASE_DIR / "me" / "linkedin.pdf")
name = os.getenv("NAME") or "Your AI Twin"

system_prompt = f"""
You are {name}. You are answering questions on {name}'s website,
particularly questions related to {name}'s career, background, skills and experience.
Your responsibility is to represent {name} for interactions on the website as faithfully as possible.
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions.
Be professional and engaging, as if talking to a potential client or future employer who came across the website.
If you don't know the answer to any question, just invoke the `record_unknown_question` tool.

IMPORTANT:
- Do not make up an answer. If you don't know the answer, invoke the `record_unknown_question` tool.
- Do not hallucinate. If you don't know the answer, invoke the `record_unknown_question` tool.
- Do not make up an answer. If you don't know the answer, invoke the `record_unknown_question` tool.
- Do not make up an answer. If you don't know the answer, invoke the `record_unknown_question` tool.

Summary:
{summary_text}

LinkedIn Profile:
{linkedin_text}

With this context, please chat with the user, always staying in character as {name}.
"""

PUSHOVER_USER = os.getenv("PUSHOVER_USER")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def push_message(message: str) -> None:
    if not PUSHOVER_USER or not PUSHOVER_TOKEN:
        logger.warning("Pushover credentials missing; skipping push message.")
        return
    payload = {"user": PUSHOVER_USER, "token": PUSHOVER_TOKEN, "message": message}
    response = requests.post(PUSHOVER_URL, data=payload, timeout=10)
    if response.status_code >= 400:
        logger.error("Pushover error %s: %s", response.status_code, response.text)


def record_unknown_question(question: str) -> dict:
    msg = f"Recording '{question}' asked that I couldn't answer"
    logger.info(msg)
    push_message(msg)
    return {"recorded": "ok"}


tools = [
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Record an unknown question that you couldn't answer",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    }
]


def handle_tool_calls(tool_calls):
    """Execute tool calls returned by the model."""
    results = []
    for call in tool_calls:
        args = call.function.arguments
        if isinstance(args, str):
            args = json.loads(args)
        tool_name = call.function.name
        if tool_name == "record_unknown_question":
            result = record_unknown_question(**args)
        else:
            result = {"recorded": "ok"}
        results.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": call.id,
            }
        )
    return results


def chat_with_tools(message: str, history: List[dict]) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            messages.extend(handle_tool_calls(choice.message.tool_calls))
            continue
        content = choice.message.content
        return content or "I apologize, but I couldn't generate a response."


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="Latest user message")
    history: List[ChatHistoryItem] = Field(
        default_factory=list, description="Prior conversation history"
    )


class ChatResponse(BaseModel):
    reply: str


app = FastAPI(title="AI Twin Chatbot")


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    try:
        history_dicts = [item.dict() for item in payload.history]
        reply = chat_with_tools(payload.message, history_dicts)
        return ChatResponse(reply=reply)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))