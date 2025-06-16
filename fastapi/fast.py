from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---- 모델 로드 (여기서 경로/토큰 수정!) ----
MODEL_PATH = "kakaocorp/kanana-1.5-2.1b-instruct-2505"  # HF 모델명 또는 로컬경로
# 본인 토큰

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, token=HF_TOKEN)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)  # GPU, CPU면 device=-1

# ---- FastAPI 앱 생성 ----
app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate(query: Query):
    result = generator(query.prompt, max_new_tokens=128)[0]['generated_text']
    return {"result": result}
