# FastAPI 및 관련 라이브러리 임포트
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai

# .env 파일에서 환경변수(OPENAI_API_KEY 등) 불러오기
load_dotenv()

# OpenAI API 클라이언트 생성 (API 키는 환경변수에서 읽음)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OpenAI GPT 모델을 사용해 텍스트 생성 함수
def generate_text(prompt):
    # OpenAI Chat API 호출 (gpt-3.5-turbo 사용)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},  # 시스템 프롬프트
            {"role": "user", "content": prompt}  # 유저 입력
        ],
        max_tokens=256,      # 최대 토큰 수
        temperature=0.7,     # 창의성(랜덤성) 조절
    )
    # 생성된 답변 텍스트 반환
    return response.choices[0].message.content.strip()

# (테스트용) 터미널에서 직접 실행 시 동작: 사용자 입력을 받아 답변 출력
if __name__ == "__main__":
    user_input = input("질문을 입력하세요: ")
    print(generate_text(user_input))


# 준혁냄 여기 넣으셈 def 형태 






# FastAPI 앱 인스턴스 생성
app = FastAPI()

# POST 요청에서 받을 데이터 구조 정의 (pydantic BaseModel 사용)
class Query(BaseModel):
    prompt: str  # 질문 프롬프트

# /generate/ 엔드포인트: POST 요청으로 프롬프트를 받아 OpenAI로 답변 생성 후 반환
@app.post("/generate/")
async def generate(query: Query):
    result = generate_text(query.prompt)
    return {"result": result}
