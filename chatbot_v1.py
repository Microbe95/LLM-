# pip install -U \
#   python-dotenv \               # .env 환경변수 로드용
#   openai \                      # OpenAI API 사용
#   langchain \                   # LangChain 핵심 라이브러리
#   langchain-community \         # JSONLoader, FAISS, Chroma 등 커뮤니티 구성요소
#   langchain-openai \            # OpenAI ↔ LangChain 연동
#   langchain-core \              # Document, Retriever 등 코어 객체
#   tiktoken \                    # 토큰 길이 측정용 (OpenAI 모델용)
#   faiss-cpu \                   # FAISS 벡터 저장소 (CPU 버전)
#   pydantic \                    # 사용자 정의 리트리버 클래스(Field) 정의용
#   jq                            # JSONLoader에서 jq 스키마 파싱용

# pip install -U python-dotenv openai langchain langchain-community langchain-openai langchain-core tiktoken faiss-cpu pydantic jq
# 패키지 인스톨 정리 본

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


# 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. JSON 로더 설정 - contents는 page_content로, date는 metadata로 이동

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


# 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

split_docs = splitter.split_documents(documents)


# 3. 임베딩 및 벡터 저장소 생성
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai.api_key
)
# - FAISS 벡터 저장소 생성
db = FAISS.from_documents(split_docs, embedding_model)


# 5. 리트리버 및 LLM 구성

# - 날짜 정렬 리트리버 (Pydantic + LangChain 최신 방식)
class DateSortedRetriever(BaseRetriever, RunnableSerializable):
    base_retriever: BaseRetriever = Field(...)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        docs_sorted = sorted(
            docs,
            key=lambda d: d.metadata.get("date", "9999-99-99").replace(".", "-")
        )
        return docs_sorted
    
# - 리트리버 + 정렬 래퍼 적용
retriever = db.as_retriever(search_kwargs={"k": 10})
sorted_retriever = DateSortedRetriever(base_retriever=retriever)



# 6. 프롬프트 템플릿 정의 및 QA 체인 구성

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
# gpt-3.5-turbo
qa_chain = RetrievalQA.from_chain_type (
    llm=llm,
    retriever=sorted_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True  
)

# 7. 히스토리 기반 챗봇 클래스 정의
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

        # 결과 출력
        print("\n" + "=" * 60)
        print("❓ 질문:")
        print(user_query)
        print("\n" + "-" * 60)
        print("🤖 답변:\n")
        print(answer)
        # print("\n" + "-" * 60)
        # print("📚 참고 문서 출처 (날짜 기준):\n")

        # for i, doc in enumerate(sources, start=1):
        #     date = doc.metadata.get("date", "N/A")
        #     type = doc.metadata.get("type", "N/A")
        #     preview = doc.page_content.strip().replace("\n", " ")[:400] + "..."  # 내용 미리보기
        #     print(f"{i}. 날짜: {date} / 내용: {preview} / 유형: {type}")

        # print("=" * 60 + "\n")
        return answer

# 8. 챗봇 객체 생성 및 테스트
chatbot = CBAMChatbot(qa_chain)

while True:
    user_input = input("\n❓ 질문을 입력하세요 (종료하려면 'exit'): ")
    if user_input.lower() in ['exit', 'quit']:
        break
    chatbot.ask(user_input)
