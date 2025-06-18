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

# --------------------- 5. 프롬프트 및 QA 체인 ---------------------
prompt_template = PromptTemplate.from_template("""
너는 지금 CBAM 플랫폼 ‘카봇’이라는 챗봇 역할을 수행하고 있어.

❗ 다음과 같은 이유로 절대 50자를 넘기면 안 돼:

1. 실무자는 핵심만 빠르게 확인하길 원해.
2. 네가 장황하게 설명하면 **잘못된 추론**이나 **과잉 안내**로 오해가 생긴다.
3. 특히 문서에 정보가 없을 경우, 너는 **추측하지 말고, 간단한 안내 한 줄**로 끝내야 신뢰를 잃지 않는다.

📌 따라서 반드시 다음 규칙을 지켜:

1. **모든 답변은 50자 이내로 작성.**
    
    → 문장 수 하나, 부연 설명 금지.
    
2. **manual = 플랫폼 반영 사항 정리 문서**
    
    👉 플랫폼이 실제로 어떤 방식으로 구현되었는지를 설명하는 문서니까 플랫폼 관련 질문이 나오면 여기서 참고해
    
3. **guide = EU 탄소국경제도 전환기간 이행 가이드라인**
    
    👉 제도의 기본적인 절차와 방식이 정리된 공식 가이드니까 CBAM 관련 한 질문은 여기서 참고해
    
4. **cbam = 알기쉽게 풀어쓴 CBAM 해설서**
    
    👉 제도의 기본적인 절차와 방식이 정리된 공식 가이드니까 CBAM 관련 한 질문은 여기서 참고해
    
5. **omnibus = EU 옴니버스 패키지 법안 요약**
    
    👉 CBAM 제도가 개정된 내용이 담겨 있으니까 CBAM 관련 질문이 들어오면 이 파일을 먼저 확인하고 대답해
    
6. **문서에 없으면 아래 문장만 그대로 출력:**

> "해당 정보는 문서에 없습니다. 시스템 담당자에게 문의해 주세요."
> 

이 문장은 **단 하나의 공식 문장**이야.

다른 표현(예: ‘관련 기관 문의’, ‘공식 웹사이트 참조’, ‘제공되지 않음’)로 바꾸면 **정책 위반**이다.

1. 사용자가 보기엔 너는 친절한 안내자지만, 내부적으로는 **절대 추측하거나 임의 판단하지 않는 도우미**야.

너는 대화형 챗봇이 아니라, **정확한 문서 기반 정보 요약기**로 작동해야 해.

한 번이라도 규칙을 어기면 **답변은 잘못된 정보로 간주되고 무효 처리된다.**

문서:
{context}

질문:
{question}
""")

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=sorted_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --------------------- 6. 챗봇 클래스 ---------------------
class CBAMChatbot:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.history = []

    def ask(self, user_query: str):

        # 프롬프트 히스토리 구성
        history_prompt = ""
        for u, b in self.history[-5:]:
            history_prompt += f"User: {u}\nBot: {b}\n"

        # 질의 수행
        result = self.qa_chain.invoke({"query": user_query})
        answer = result["result"]   
        sources = result["source_documents"]    

        # 히스토리 저장
        self.history.append((user_query, answer))
# class CBAMChatbot:
#     def __init__(self, qa_chain):
#         self.qa_chain = qa_chain
#         self.history = []

#     def ask(self, user_query: str):
#         result = self.qa_chain.invoke({"query": user_query})
#         answer = result["result"]
#         sources = result["source_documents"]
#         self.history.append((user_query, answer))
#         return {
#             "answer": answer,
#             "sources": [
#                 {
#                     "date": doc.metadata.get("date", "N/A"),
#                     "preview": doc.page_content[:]
#                 }
#                 for doc in sources
#             ]
#         }

chatbot = CBAMChatbot(qa_chain)

# --------------------- 7. 질문 로딩 ---------------------
import pandas as pd
question_df = pd.read_excel("Q-list.xlsx", header=2)
questions = [q for q in question_df["Unnamed: 3"].dropna().tolist() if q != "질문"]
# 정답 추출 (숫자, 이상한 값 제외)
ground_truths = [
    g for g in question_df["Unnamed: 7"]
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
        "ground_truth": g   # ✅ 여기에 실제 정답 입력
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

