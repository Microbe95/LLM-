# 1. 라이브러리 설치
# !pip install openai langchain tiktoken pandas openpyxl


# 3. chatbot 불러오기
from chatbot_v1 import CBAMChatbot, qa_chain# 챗봇 클래스를 실제 클래스로 변경하세요

# 챗봇 인스턴스 생성
chatbot = CBAMChatbot(qa_chain)

# 4. 질문 데이터 불러오기
import pandas as pd

# 업로드된 Excel 파일 로딩 
excel_file = "C:\\Users\\bitcamp\\Desktop\\a\\FAQ_예상 질문 추출 리스트.xlsx"# 경로를 실제 파일 경로로 변경하세요
df = pd.read_excel(excel_file)

# 질문 열 이름 확인
print(df.columns)

# 예: "질문"이라는 컬럼명이 존재한다면
questions = df["질문"].dropna().tolist()

# 5. 질문 반복 → 답변 생성
results = []
for question in questions:
    try:
        response = chatbot.ask(question)
        results.append({
            "질문": question,
            "답변": response["answer"],
            "출처(날짜)": ", ".join([s["date"] for s in response["sources"]])
        })
    except Exception as e:
        results.append({
            "질문": question,
            "답변": f"오류 발생: {e}",
            "출처(날짜)": ""
        })

# 6. 결과 저장
results_df = pd.DataFrame(results)

# CSV 저장
results_df.to_csv("챗봇_응답_결과.csv", index=False)

# JSON 저장
results_df.to_json("챗봇_응답_결과.json", force_ascii=False, orient="records", indent=2)

