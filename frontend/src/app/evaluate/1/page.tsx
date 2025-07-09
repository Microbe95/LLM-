// ✅ EvaluatePage.tsx - 개선된 탭 UI 복원 및 통합 표시
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { markStepCompleteAuto } from "@/utils/stepTracker";
import OverviewTab from "@/components/evaluate/OverviewTab";
import IssueTab from "@/components/evaluate/IssueTab";
import StakeholderTab from "@/components/evaluate/StakeholderTab";

export default function EvaluatePage() {
  const router = useRouter();
  const [tab, setTab] = useState("overview");

  useEffect(() => {
    markStepCompleteAuto();
  }, []);

  const handleNext = () => {
    router.push("/mapping/1");
  };

  const summaryCards = [
    { label: "총 발송", unit: "(회)", value: null },
    { label: "총 응답", unit: "(회)", value: null },
    { label: "응답률", unit: "(%)", value: null },
  ];

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">분석 및 평가</h2>
        <StepBar current="Evaluate" />

        {/* 요약 카드 */}
        <section className="flex gap-4 my-6 max-w-6xl mx-auto">
          {summaryCards.map((card, i) => (
            <div key={i} className="flex-1 text-center border rounded p-4 bg-gray-50">
              <p className="text-sm text-gray-600">{card.label}</p>
              <p className="text-2xl font-bold">{card.value ?? "-"} {card.unit}</p>
            </div>
          ))}
        </section>

        {/* 탭 UI */}
        <section className="max-w-6xl mx-auto border rounded overflow-hidden">
          <nav className="flex bg-gray-100">
            {[
              { key: "overview", label: "개요" },
              { key: "issue", label: "이슈 중요도 분석" },
              { key: "stakeholder", label: "이해관계자별 분석" },
            ].map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                className={`flex-1 px-4 py-2 text-sm font-medium transition-all duration-200 ${
                  tab === key ? "bg-white border-b-2 border-blue-600 text-blue-700" : "text-gray-600 hover:bg-gray-200"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>

          <div className="bg-white p-6">
            {tab === "overview" && <OverviewTab />}
            {tab === "issue" && <IssueTab />}
            {tab === "stakeholder" && <StakeholderTab />}
          </div>
        </section>

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
