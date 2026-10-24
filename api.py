"""Optional FastAPI service wrapper."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from chatbot import LeadChatbot

app=FastAPI(title="Ashcombe & Rowe Lead Qualification Chatbot", version="1.0.0")
bot=LeadChatbot()

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)

class ChatResponse(BaseModel):
    session_id: str
    response: str

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    try:
        return ChatResponse(session_id=body.session_id, response=bot.reply(body.session_id, body.message))
    except Exception as exc:
        # Do not leak exception details or user data.
        raise HTTPException(status_code=503, detail="The chatbot is temporarily unavailable.") from exc
