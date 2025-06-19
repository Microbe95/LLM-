from datasets import Dataset
from dotenv import load_dotenv
import os
import openai
import pandas as pd
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import TokenTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from langchain_core.runnables import RunnableSerializable
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    # context_entities_recall,
)
from ragas import evaluate

from typing import List

# --------------------- 1. 환경 설정 ---------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --------------------- 2. 문서 로딩 및 분할 ---------------------
# 1. JSON 로더 설정 - contents는 page_content로, date는 metadata로 이동
loader = JSONLoader(
    file_path="./merged_cbam_data.json",
    jq_schema=".[]",                    # JSON 배열 구조
    text_content=True,                 # contents → page_content
    content_key="contents",            # page_content 필드 지정 (만약 "text"면 바꿔야 함)
    metadata_func=lambda record, _: {
        "date": record.get("date", "unknown"),
        "type": record.get("type", "unknown")  # type 필드도 포함
    }
)

documents = loader.load()
# 2. Text Splitter 설정
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# --------------------- 3. 벡터 저장소 ---------------------
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai.api_key
)
db = FAISS.from_documents(split_docs, embedding_model)

# --------------------- 4. 리트리버 구성 ---------------------
class DateSortedRetriever(BaseRetriever, RunnableSerializable):
    base_retriever: BaseRetriever = Field(...)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        docs_sorted = sorted(
            docs,
            key=lambda d: d.metadata.get("date", "9999-99-99").replace(".", "-")
        )
        return docs_sorted

retriever = db.as_retriever(search_kwargs={"k": 10})
sorted_retriever = DateSortedRetriever(base_retriever=retriever)

# --------------------- 5. 프롬프트 및 QA 체인 --------------------- 짧은 답변 프롬프트
prompt_template = PromptTemplate.from_template("""
                                               
너는 지금 CBAM 플랫폼에서 ‘카봇’이라는 친절한 챗봇 역할을 수행하고 있어.

❗ 다음과 같은 이유로 200자를 넘기면 안 돼:

1. 실무자는 핵심만 빠르게 확인하길 원해.  
2. 네가 장황하게 설명하면 **잘못된 추론**이나 **과잉 안내**로 오해가 생긴다.  
3. 특히 문서에 정보가 없을 경우, 너는 **추측하지 말고, 간단한 안내 한 줄**로 끝내야 신뢰를 잃지 않는다.
                                               
---

📌 따라서 반드시 다음 규칙을 지켜:

1. **모든 답변은 200자 이내로 작성.**  
   → 문장 5문장 이하, 부연 설명 금지.

2. **manual = 플랫폼 반영 사항 정리 문서**  
   → 플랫폼이 실제로 어떤 방식으로 구현되었는지를 설명하는 문서니까 플랫폼 관련 질문이 나오면 여기서 참고해

3. **guide = EU 탄소국경제도 전환기간 이행 가이드라인**  
   → 제도의 기본적인 절차와 방식이 정리된 공식 가이드니까 CBAM 관련 질문은 여기서 참고해

4. **cbam = 알기쉽게 풀어쓴 CBAM 해설서**  
   → 제도의 기본적인 절차와 방식이 정리된 공식 가이드니까 CBAM 관련 질문은 여기서 참고해
                                               
5. **문서에 없으면 아래 문장만 그대로 출력:**
   → 해당 정보는 제가 답변드리기 어려운 내용입니다. CBAM 공식 기관 혹은 시스템 담당자에게 문의해 주세요.
   - 이 문장은 **단 하나의 공식 문장**이야.  
   - 다른 표현(예: ‘관련 기관 문의’, ‘공식 웹사이트 참조’, ‘제공되지 않음’)로 바꾸면 **정책 위반**이다.

6. 사용자가 보기엔 너는 친절한 안내자지만, 내부적으로는 **절대 추측하거나 임의 판단하지 않는 도우미**야.
   - 너는 대화형 챗봇이 아니라, **정확한 문서 기반 정보 요약기**로 작동해야 해.
   - 한 번이라도 규칙을 어기면 **답변은 잘못된 정보로 간주되고 무효 처리된다.**
                                               
7. **말투는 "입니다/어요" 혼합체를 사용**
   - 격식은 있지만 부담 없는 어투를 사용하도록 해.
                                               
문서:
{context}

질문:
{question}
""")
# ----------------------------------------------------------------------------------- #  긴답변 프롬포트 
# prompt_template = PromptTemplate.from_template("""
# 안녕하세요. 저는 '카봇'이에요.  
# CBAM(탄소국경조정제도) 대응 플랫폼을 직접 개발했고, 지금은 이 플랫폼과 관련된 내용을 안내해드리고 있어요.

# 제가 참고하는 문서는 다음과 같이 서로 다른 역할을 해요:
                            
# 1. **manual = 플랫폼 반영 사항 정리 문서**  
#    👉 플랫폼이 실제로 어떤 방식으로 구현되었는지를 설명하는 문서예요.  
#    👉 '원래는 이렇게 해야 하는데, 우리 플랫폼에서는 이렇게 구현했어요'처럼 비교 중심으로 구성되어 있어요.

# 2. **guide=  EU 탄소국경제도 전환기간 이행 가이드라인**  
#    👉 제도의 기본적인 절차와 방식이 정리된 공식 가이드예요.

# 3. **cbam= 알기쉽게 풀어쓴 CBAM 해설서**  
#    👉 제도 전체의 개념과 용어를 쉽게 설명한 자료예요.

# 4. **omnibus  = EU 옴니버스 패키지 법안 요약**  
#    👉 CBAM 제도가 개정된 내용이 담겨 있고, 일부는 플랫폼에도 반영되어 있어요.

# ---

# 📌 **답변 기준은 아래와 같아요:**

# - ❗ 질문이 제도 전반에 대한 개념을 묻는 경우에는 **가이드라인이나 해설서**를 참고해서 개념을 설명할게요.
# - ❗ 질문이 '우리 플랫폼에서 어떻게 되나요?'처럼 구현 방식을 묻는 경우에는 **플랫폼 반영 사항 정리 문서**를 중심으로 설명드릴게요.
# - ❗ 서로 다른 문서 간 내용이 상이할 경우, **플랫폼 구현 기준을 우선** 따르되, 필요하면 차이점도 함께 알려드릴게요.
# - ❌ 문서에 없는 내용은 답변드릴 수 없으며, 이 경우 이렇게 안내할게요: **"문서에서 해당 정보를 찾을 수 없습니다."**

# 저는 실무자가 이해하기 쉽게, 질문의 맥락에 따라 개념과 플랫폼 구현을 구분해서 설명해드릴 수 있어요.  
# 궁금한 점이 있다면 언제든지 편하게 질문해주세요!


# 문서:
# {context}

# 질문:
# {question}
# """)

# ----------------------------------------------------------------------------------- # 


llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=sorted_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --------------------- 6. 챗봇 클래스 ---------------------(학습용)
# class CBAMChatbot:
#     def __init__(self, qa_chain):
#         self.qa_chain = qa_chain
#         self.history = []

#     def ask(self, user_query: str):

#         # 프롬프트 히스토리 구성
#         history_prompt = ""
#         for u, b in self.history[-5:]:
#             history_prompt += f"User: {u}\nBot: {b}\n"

#         # 질의 수행
#         result = self.qa_chain.invoke({"query": user_query})
#         answer = result["result"]   
#         sources = result["source_documents"]    

#         # 히스토리 저장
#         self.history.append((user_query, answer))
# ------------------------------------------(학습용)

# --------------------- 6. 챗봇 클래스 ---------------------(실제 서비스용)
class CBAMChatbot:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.history = []

    def ask(self, user_query: str):
        result = self.qa_chain.invoke({"query": user_query})
        answer = result["result"]
        sources = result["source_documents"]
        self.history.append((user_query, answer))
        return {
            "answer": answer,
            "sources": [
                {
                    "date": doc.metadata.get("date", "N/A"),
                    "preview": doc.page_content[:]
                }
                for doc in sources
            ]
        }

chatbot = CBAMChatbot(qa_chain)
# ------------------------------------------(실제 서비스용)


# --------------------- 7. 질문 로딩 ---------------------
import pandas as pd
question_df = pd.read_excel("Q-list_평가용최종.xlsx", header=2)
questions = [q for q in question_df["Unnamed: 3"].dropna().tolist() if q != "질문"]
# 정답 추출 (숫자, 이상한 값 제외)
ground_truths = [
    g for g in question_df["Unnamed: 8"]
    if pd.notna(g)
    and isinstance(g, str)  # 문자열만 허용
    and len(g.strip()) > 5  # 너무 짧은 건 제외
    and g.strip() != "대답"
]
# --------------------- 8. RAGAS 평가용 데이터셋 생성 ---------------------
samples = []
for q, g in zip(questions, ground_truths):
    response = chatbot.ask(q)
    samples.append({
        "question": q,
        "answer": response["answer"],
        "contexts": [ctx["preview"] for ctx in response["sources"]],
        "ground_truth": g   # 여기에 실제 정답 입력
    })

ragas_dataset = Dataset.from_list(samples)

# --------------------- 9. RAGAS 질문별 평가 실행 ---------------------
from ragas import SingleTurnSample, EvaluationDataset

for idx, sample_dict in enumerate(samples):
    sample = SingleTurnSample(
        user_input=sample_dict["question"],
        retrieved_contexts=sample_dict["contexts"],
        response=sample_dict["answer"],
        reference=sample_dict["ground_truth"]
    )
    dataset = EvaluationDataset(samples=[sample])
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    print(f"\n🔍 Q{idx+1}: {sample.user_input}")
    print(f"💬 챗봇 응답: {sample.response}")
    print(result)

