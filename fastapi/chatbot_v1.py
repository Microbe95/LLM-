
from dotenv import load_dotenv
import os
import openai
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_core.retrievers import BaseRetriever
from typing import List
from dataclasses import dataclass
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import Field
from langchain_core.runnables import RunnableSerializable
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma

# 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. JSON 로더 설정 - contents는 page_content로, date는 metadata로 이동
loader = JSONLoader(
    file_path="./merged_cbam_data.json",
    jq_schema=".[]",                   # JSON 배열 구조
    text_content=True,                # contents → page_content
    content_key="contents",           # page_content 필드 지정
    metadata_func=lambda record, _: {"date": record.get("date", "unknown")} 
)

documents = loader.load()

# 2. Text Splitter 설정

splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

split_docs = splitter.split_documents(documents)


# 4. 임베딩 및 벡터 저장소 생성
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai.api_key
)

# 5. FAISS 벡터 저장소 생성
db = Chroma.from_documents(split_docs, embedding_model, persist_directory="./chroma_db")
retriever = db.as_retriever(search_kwargs={"k": 10})
# 6. 리트리버 및 LLM 구성

# ✅ 날짜 정렬 리트리버 (Pydantic + LangChain 최신 방식)
class DateSortedRetriever(BaseRetriever, RunnableSerializable):
    base_retriever: BaseRetriever = Field(...)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        docs_sorted = sorted(
            docs,
            key=lambda d: d.metadata.get("date", "9999-99-99").replace(".", "-")
        )
        return docs_sorted
# ✅ 리트리버 + 정렬 래퍼 적용
retriever = db.as_retriever(search_kwargs={"k": 10})
sorted_retriever = DateSortedRetriever(base_retriever=retriever)



# 7. 프롬프트 템플릿 정의 및 QA 체인 구성

prompt_template = PromptTemplate.from_template("""
안녕하세요. 저는 '카봇'이에요.  
CBAM(탄소국경조정제도) 대응 플랫폼을 직접 개발했고, 지금은 이 플랫폼과 관련된 내용을 안내해드리고 있어요.

제가 참고하는 문서는 다음과 같이 서로 다른 역할을 해요:
                            
1. **플랫폼 반영 사항 정리 문서**  
   👉 플랫폼이 실제로 어떤 방식으로 구현되었는지를 설명하는 문서예요.  
   👉 '원래는 이렇게 해야 하는데, 우리 플랫폼에서는 이렇게 구현했어요'처럼 비교 중심으로 구성되어 있어요.

2. **EU 탄소국경제도 전환기간 이행 가이드라인**  
   👉 제도의 기본적인 절차와 방식이 정리된 공식 가이드예요.

3. **알기쉽게 풀어쓴 CBAM 해설서**  
   👉 제도 전체의 개념과 용어를 쉽게 설명한 자료예요.

4. **EU 옴니버스 패키지 법안 요약**  
   👉 CBAM 제도가 개정된 내용이 담겨 있고, 일부는 플랫폼에도 반영되어 있어요.

---

📌 **답변 기준은 아래와 같아요:**

- ❗ 질문이 제도 전반에 대한 개념을 묻는 경우에는 **가이드라인이나 해설서**를 참고해서 개념을 설명할게요.
- ❗ 질문이 '우리 플랫폼에서 어떻게 되나요?'처럼 구현 방식을 묻는 경우에는 **플랫폼 반영 사항 정리 문서**를 중심으로 설명드릴게요.
- ❗ 서로 다른 문서 간 내용이 상이할 경우, **플랫폼 구현 기준을 우선** 따르되, 필요하면 차이점도 함께 알려드릴게요.
- ❌ 문서에 없는 내용은 답변드릴 수 없으며, 이 경우 이렇게 안내할게요: **"문서에서 해당 정보를 찾을 수 없습니다."**

저는 실무자가 이해하기 쉽게, 질문의 맥락에 따라 개념과 플랫폼 구현을 구분해서 설명해드릴 수 있어요.  
궁금한 점이 있다면 언제든지 편하게 질문해주세요!


문서:
{context}

질문:
{question}
""")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=2000)
qa_chain = RetrievalQA.from_chain_type (
    llm=llm,
    retriever=sorted_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True  
)

# 8. 히스토리 기반 챗봇 클래스 정의
class CBAMChatbot:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.history = []

    def ask(self, user_query: str):
        # # 프롬프트 히스토리 구성
        # history_prompt = ""
        # for u, b in self.history[-5:]:
        #     history_prompt += f"User: {u}\nBot: {b}\n"

        # 질의 수행
        result = self.qa_chain({"query": user_query})
        answer = result["result"]
        sources = result["source_documents"]

        # 히스토리 저장
        self.history.append((user_query, answer))

        return {
            "answer": answer,
            "sources": [
                {
                    "date": doc.metadata.get("date", "N/A"),
                    "preview": doc.page_content[:100]
                }
                for doc in sources
            ]
        }

__all__ = ["CBAMChatbot", "qa_chain"]