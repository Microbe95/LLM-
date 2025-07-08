"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import OverviewTab from "@/components/evaluate/OverviewTab";
import IssueTab from "@/components/evaluate/IssueTab";
import StakeholderTab from "@/components/evaluate/StakeholderTab";
import { markStepCompleteAuto } from "@/utils/stepTracker"; // ✅ 추가

export default function EvaluatePage() {
  const [tab, setTab] = useState("overview");
  const router = useRouter();

  // ✅ 페이지 진입 시 step3 자동 저장
  useEffect(() => {
    markStepCompleteAuto();
  }, []);

  const handleNext = () => {
    // 추후 데이터 저장 로직 추가 가능
    router.push("/mapping/1");
  };

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">분석 및 평가</h2>
        <StepBar current="Evaluate" />

        {/* 상단 요약 카드 */}
        <div className="flex gap-4 my-6 max-w-6xl mx-auto">
          {[{ label: "총 발송", unit: "(회)" }, { label: "총 응답", unit: "(회)" }, { label: "응답률", unit: "(%)" }].map(
            (card, i) => (
              <div key={i} className="flex-1 text-center border rounded p-4 bg-gray-50">
                <p className="text-sm text-gray-600">{card.label}</p>
                <p className="text-2xl font-bold">0 {card.unit}</p>
              </div>
            )
          )}
        </div>

        {/* 탭 */}
        <div className="max-w-6xl mx-auto border rounded">
          <div className="flex bg-blue-100 rounded-t overflow-hidden">
            <button
              onClick={() => setTab("overview")}
              className={`flex-1 px-4 py-2 text-sm font-medium ${
                tab === "overview" ? "bg-white border-b-0" : "border-b"
              }`}
            >
              개요
            </button>
            <button
              onClick={() => setTab("issue")}
              className={`flex-1 px-4 py-2 text-sm font-medium ${
                tab === "issue" ? "bg-white border-b-0" : "border-b"
              }`}
            >
              이슈 별 중요도 분석
            </button>
            <button
              onClick={() => setTab("stakeholder")}
              className={`flex-1 px-4 py-2 text-sm font-medium ${
                tab === "stakeholder" ? "bg-white border-b-0" : "border-b"
              }`}
            >
              이해 관계자별 응답현황 상세 분석
            </button>
          </div>

          <div className="bg-white p-6">
            {tab === "overview" && <OverviewTab />}
            {tab === "issue" && <IssueTab />}
            {tab === "stakeholder" && <StakeholderTab />}
          </div>
        </div>

        <div className="flex justify-end max-w-6xl mx-auto mt-6">
          <button
            onClick={handleNext}
            className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
          >
            저장하고 넘어가기
          </button>
        </div>
      </div>
    </main>
  );
}
