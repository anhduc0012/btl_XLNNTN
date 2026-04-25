import sys
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque

# Đảm bảo import được các module local
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.intent_classifier import get_intent
from backend.nlp_engine import engine

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class Memory:
    def __init__(self, limit=5):
        self.history = deque(maxlen=limit) # Lưu tối đa 5 câu gần nhất
        self.last_disease = None

    def add(self, user_msg, bot_msg):
        self.history.append({"user": user_msg, "bot": bot_msg})

    def get_context(self):
        return " ".join([h["user"] for h in self.history])

sessions = {}

def get_memory(session_id: str) -> Memory:
    if session_id not in sessions:
        sessions[session_id] = Memory()
    return sessions[session_id]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    user_msg = request.message
    session_id = request.session_id
    memory = get_memory(session_id)
    
    # 1. Nhận diện ý định (Chuẩn chỉ)
    intent = get_intent(user_msg)
    
    # 2. Xử lý theo ý định
    response_data = {}
    
    if intent == "chit_chat":
        chat_history = memory.get_context()
        ans = engine.handle_chit_chat(user_msg, chat_history)
        response_data = {"qa": ans}
        
    elif intent == "task_location":
        response_data = engine.execute_task_location(user_msg)
        
    elif intent == "nutrition_query":
        response_data = engine.handle_nutrition_query(user_msg)

    elif intent == "symptom_query" or intent == "medical_query":
        # Kết hợp ngữ cảnh từ bộ nhớ nếu câu hỏi quá ngắn
        context_msg = user_msg
        if len(user_msg.split()) < 4 and memory.last_disease:
            context_msg = f"{user_msg} liên quan đến {memory.last_disease}"
            
        chat_history = memory.get_context()
        response_data = engine.handle_medical_query(context_msg, chat_history)
        if "disease" in response_data:
            memory.last_disease = response_data["disease"]

    # 3. Lưu vào bộ nhớ
    memory.add(user_msg, response_data.get("qa", ""))
    
    print(f"DEBUG: Intent='{intent}', Context='{memory.last_disease}'")
    return response_data

from fastapi.staticfiles import StaticFiles
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8080, reload=True)
