# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot_v1 import CBAMChatbot, qa_chain 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Next.js 개발용이라면 "*" 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 스키마
class PromptRequest(BaseModel):
    prompt: str

# 응답 스키마
class PromptResponse(BaseModel):
    result: str

# 챗봇 인스턴스
chatbot = CBAMChatbot(qa_chain)

@app.post("/generate/", response_model=PromptResponse)
async def generate_answer(payload: PromptRequest):
    result = chatbot.ask(payload.prompt)
    return PromptResponse(result=result["answer"])
