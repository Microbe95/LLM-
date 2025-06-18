"use client";
import { useRef, useEffect, useState } from "react";
import Image from "next/image";

const FAQS = [
  "CBAM 개념",
  "CN코드 확인 방법",
  "전력 배출계수는 자동으로 입력되나요?",
  "인증서 단가는 어떤 기준으로 계산되나요?",
  "산화계수란 무엇이고, 왜 적용하나요?"
];

export default function ChatbotModal({ onClose }) {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(true); // 초기 로딩 여부
  const [input, setInput] = useState("");
  const scrollRef = useRef();


  useEffect(() => {
  if (scrollRef.current) {
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }
}, [messages]);
  // ✅ localStorage에서 대화 기록 불러오기
  useEffect(() => {
    const saved = localStorage.getItem("cbam_chat_history");
    if (saved) {
      try {
        setMessages(JSON.parse(saved));
      } catch {
        localStorage.removeItem("cbam_chat_history");
        setMessages([defaultWelcomeMessage()]);
      }
    } else {
      setMessages([defaultWelcomeMessage()]);
    }
    setIsLoading(false); // 로딩 완료
  }, []);

  // ✅ 대화 기록 저장
  useEffect(() => {
    if (!isLoading) {
      localStorage.setItem("cbam_chat_history", JSON.stringify(messages));
    }
  }, [messages]);

const handleResetChat = () => {
  localStorage.removeItem("cbam_chat_history");
  setMessages([defaultWelcomeMessage()]);
};
  
  // ✅ 기본 인삿말 함수
  const defaultWelcomeMessage = () => ({
    role: "system",
    text: "안녕하세요! 저는 CBAM 궁금증을 해결해드리는 카봇이에요.\n'CBAM이 뭐야?'부터 '인증서 가격 어떻게 계산하지?'까지, 편하게 질문해주세요!"
  });

  // 질문 전송
  const sendMessage = async (question) => {
    if (!question.trim()) return;
    setMessages(prev => [
      ...prev,
      { role: "user", text: question, time: now() }
    ]);
    setInput("");

  try {
    const res = await fetch("/api/chatbot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: question }),
    });
    if (!res.ok) throw new Error("서버 응답 실패");
    const data = await res.json();
    setMessages(prev => [
      ...prev,
      { role: "bot", text: data.result, time: now() }
    ]);
  } catch (err) {
    setMessages(prev => [
      ...prev,
      { role: "bot", text: "⚠️ 서버와 통신에 실패했습니다.", time: now() }
    ]);
  }
};

  // FAQ 버튼 클릭
  const handleFAQ = (text) => {
    sendMessage(text);
  };

  // 직접 입력
  const handleSend = (e) => {
    e.preventDefault();
    sendMessage(input);
  };

  if (isLoading) return null;
  return (
    <div className="fixed z-50 bottom-0 right-0 mb-6 mr-6 md:mb-10 md:mr-10 max-w-lg w-[150vw] sm:w-[700px] bg-white rounded-2xl shadow-2xl flex flex-col">

      {/* 상단 - 로고/타이틀/설명/닫기 */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-white rounded-t-2xl">
        <div className="flex items-center space-x-3">
          {/* 로고만 이미지 */}
          <div className="w-12 h-12 rounded-full bg-[#4273C4] flex items-center justify-center shadow-md cursor-pointer"
            onClick={handleResetChat} 
            >
            <Image
              src="/LogoC.png"
              alt="챗봇 로고"
              width={60}
              height={60}
              priority
            />
          </div>
          <div>
            <div className="text-[#00235B] text-lg font-bold">당신의 CBAM 파트너, CABOT</div>
            <div className="text-sm text-gray-500 font-medium">CBAM 챗봇 상담 도우미</div>
          </div>
        </div>
        <button
          className="text-gray-400 hover:text-gray-700 text-2xl ml-3"
          onClick={onClose}
          aria-label="챗봇 닫기"
        >
          &times;
        </button>
      </div>
      {/* 안내 멘트 */}
      <div className="px-4 py-2 text-center text-sm text-[#00235B] border-b bg-[#f8fafd]">
        안녕하세요! 저는 CBAM 궁금증을 해결해드리는 카봇이에요.<br />
        "CBAM이 뭐야?"부터 "인증서 가격 어떻게 계산하지?"까지, 편하게 질문해주세요!
      </div>
      {/* FAQ 버튼 */}
      <div className="flex flex-wrap gap-2 px-4 py-2 bg-[#f8fafd]">
        {FAQS.map((faq, i) => (
          <button
            key={i}
            className="rounded-full border border-[#00235B] text-[#00235B] text-xs px-3 py-1 bg-white hover:bg-[#00235B] hover:text-white transition"
            onClick={() => handleFAQ(faq)}
          >
            {faq}
          </button>
        ))}
      </div>
      {/* 대화창 */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-3 py-3 bg-[#f8fafd]"
        style={{ maxHeight: "60vh" }}
      >
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex mb-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {/* bot/system이면 왼쪽에 로고 */}
            {msg.role !== "user" && (
              <div className="flex-shrink-0">
                <div className="w-8 h-8 rounded-full bg-[#4273C4] flex items-center justify-center mr-2 shadow">
                  <Image
                    src="/chatbot1.png"
                    alt="bot"
                    width={45}
                    height={45}
                  />
                </div>
              </div>
            )}
            <div
              className={`rounded-2xl px-4 py-2 shadow 
                ${msg.role === "user"
                  ? "bg-[#00235B] text-white ml-10"
                  : "bg-white text-gray-800 mr-10 border border-gray-100"}`}
              style={{ maxWidth: "80%" }}
            >
              <span className="whitespace-pre-line">{msg.text}</span>
              {msg.time && (
                <div className="text-xs text-gray-400 mt-1 text-right">{msg.time}</div>
              )}
            </div>
          </div>
        ))}
      </div>
      {/* 입력창 */}
      <form onSubmit={handleSend} className="flex px-3 py-2 border-t bg-white">
        <input
          className="flex-1 rounded-l-xl border border-gray-200 px-3 py-2 text-sm outline-none focus:ring"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="질문을 입력하세요"
        />
        <button
          type="submit"
          className="rounded-r-xl bg-[#00235B] text-white px-4 py-2 text-sm font-semibold hover:bg-[#003580] transition"
        >
          보내기
        </button>
      </form>
    </div>
  );
}

// 시간 포맷
function now() {
  const d = new Date();
  return `${d.getFullYear().toString().slice(2)}.${(d.getMonth() + 1)
    .toString()
    .padStart(2, "0")}.${d.getDate().toString().padStart(2, "0")} / ${d
    .getHours()
    .toString()
    .padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}

