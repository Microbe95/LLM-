// ✅ /app/survey/main/page.tsx (통합 시안 반영: 생성 버튼 포함, 설문 요약 + 응답 처리 + 설명 추가 + 발송 모달 + 선택 그룹 초기값 연동)
"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import EmailSetModal from "@/components/EmailSetModal";
import ConfirmModal from "@/components/ConfirmModal";

const SURVEY_TARGETS = [
  { label: "임직원", description: "내부 구성원", tag: "내부" },
  { label: "투자자", description: "설명", tag: "외부" },
  { label: "지역사회", description: "설명", tag: "외부" },
  { label: "전문가", description: "설명", tag: "외부" },
  { label: "고객", description: "제품/서비스 이용자", tag: "외부" },
  { label: "공급업체", description: "설명", tag: "내부" },
  { label: "정부/규제기관", description: "설명", tag: "외부" }
];

export default function SurveyMainPage() {
  const router = useRouter();
  const [sessionUser, setSessionUser] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState<string>("");
  const [questions, setQuestions] = useState<string[]>([]);
  const [title, setTitle] = useState("");
  const [group, setGroup] = useState("");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [description, setDescription] = useState("");
  const [responseOption, setResponseOption] = useState("");

  useEffect(() => {
    const user = localStorage.getItem("sessionUser");
    if (!user) {
      alert("로그인이 필요합니다.");
      router.push("/login");
    }
    setSessionUser(user);
  }, [router]);

  const handleGenerateSurvey = () => {
    setQuestions([
      "ESG 중대성 항목을 평가해주세요",
      "해당 기업의 지속 가능성에 대한 의견을 작성해주세요"
    ]);
  };

  const handleConfirmSend = () => {
    if (!selectedTarget) {
      alert("먼저 설문 대상자를 선택해주세요.");
      return;
    }
    setShowConfirm(true);
  };

  const handleConfirmSubmit = () => {
    alert(`${selectedTarget} 그룹에게 설문이 발송되었습니다.`);
    setShowConfirm(false);
  };

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">설문조사 설정</h2>
        <StepBar current="Survey" />

        <div className="grid grid-cols-2 gap-6 mt-6 max-w-6xl mx-auto">
          <div className="col-span-1 space-y-4">
            <div className="border p-4 rounded bg-gray-50">
              <h3 className="font-semibold mb-2">설문조사 기본 정보</h3>
              <label className="block text-sm mb-1">설문 제목</label>
              <input value={title} onChange={(e) => setTitle(e.target.value)} className="w-full border px-3 py-2 rounded mb-2" />

              <label className="block text-sm mb-1">이해관계자 구분</label>
              <input value={group} onChange={(e) => setGroup(e.target.value)} className="w-full border px-3 py-2 rounded mb-2" />

              <label className="block text-sm mb-1">설문 기간</label>
              <div className="flex gap-2">
                <input type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} className="border px-3 py-2 rounded w-full" />
                <input type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} className="border px-3 py-2 rounded w-full" />
              </div>
            </div>

            <div className="border p-4 rounded bg-gray-50">
              <div className="flex justify-between mb-2">
                <h3 className="font-semibold">설문조사 대상 선택</h3>
                <button onClick={() => setShowModal(true)} className="text-sm text-blue-600 border px-2 py-1 rounded hover:bg-blue-100">Email 설정하기</button>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {SURVEY_TARGETS.map((target) => (
                  <label
                    key={target.label}
                    className={`flex flex-col border rounded px-3 py-2 cursor-pointer ${
                      selectedTarget === target.label ? "border-blue-600 bg-white" : "border-gray-300 bg-gray-100"
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <div className="font-medium">{target.label}</div>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        target.tag === "내부" ? "bg-blue-100 text-blue-600" : "bg-gray-200 text-gray-700"
                      }`}>
                        {target.tag}
                      </span>
                    </div>
                    <div className="text-sm text-gray-500 mt-1">{target.description}</div>
                    <input type="radio" name="survey-target" className="hidden" checked={selectedTarget === target.label} onChange={() => setSelectedTarget(target.label)} />
                  </label>
                ))}
              </div>
            </div>

            <div className="border p-4 rounded bg-gray-50">
              <h3 className="font-semibold mb-2">불성실 응답 처리 방법 설정</h3>
              <div className="space-y-2 text-sm">
                <label className="block"><input type="radio" name="responseOption" value="지시적" onChange={(e) => setResponseOption(e.target.value)} className="mr-2" />지시적 조작 점검 문항</label>
                <label className="block"><input type="radio" name="responseOption" value="가짜문항" onChange={(e) => setResponseOption(e.target.value)} className="mr-2" />가짜문항</label>
                <label className="block"><input type="radio" name="responseOption" value="자기보고" onChange={(e) => setResponseOption(e.target.value)} className="mr-2" />자기 보고식 노력 측정 문항</label>
              </div>
            </div>

            <div className="border p-4 rounded bg-gray-50">
              <h3 className="font-semibold mb-2">설문예시</h3>
              <p className="text-sm text-gray-700">
                ESG 중대성 평가 설문조사는<br />
                귀하의 의견을 수렴해서 ESG 관점에서 중요한 항목을 알고자 합니다.
              </p>
            </div>
          </div>

          <div className="col-span-1 border p-4 rounded bg-gray-50 h-fit">
            <h3 className="font-semibold text-lg mb-3">설문 요약</h3>
            <div className="text-sm mb-4">
              <div><strong>제목:</strong> {title || "-"}</div>
              <div><strong>이해관계자:</strong> {group || "-"}</div>
              <div><strong>기간:</strong> {fromDate || "-"} ~ {toDate || "-"}</div>
            </div>

            <label className="block text-sm font-medium mb-1">설문 설명</label>
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} className="w-full h-24 border px-3 py-2 rounded mb-4" />

            <div className="flex gap-2 mb-4">
              <button onClick={handleGenerateSurvey} className="w-1/2 bg-blue-500 text-white py-2 rounded hover:bg-blue-600">설문 생성하기</button>
              <button onClick={handleConfirmSend} className="w-1/2 bg-green-600 text-white py-2 rounded hover:bg-green-700">설문 발송하기</button>
            </div>

            <h4 className="font-semibold mb-2">생성된 설문 문항</h4>
            {questions.length > 0 ? (
              <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                {questions.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-gray-500">설문 생성 버튼을 눌러 문항을 생성하세요.</p>
            )}

            <button
              onClick={() => router.push("/survey/1")}
              className="w-full mt-6 bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
            >
              저장하고 넘어가기
            </button>
          </div>
        </div>

        {showModal && <EmailSetModal onClose={() => setShowModal(false)} />}
        {showConfirm && (
          <ConfirmModal
            title="설문 발송 확인"
            message={`선택된 대상자 "${selectedTarget}"에게 설문을 발송하시겠습니까?`}
            onConfirm={handleConfirmSubmit}
            onCancel={() => setShowConfirm(false)}
          />
        )}
      </div>
    </main>
  );
}
