// ✅ 개선된 ReportPage.tsx - 보고서 초안 생성 + 수정 + PDF 출력 + 버튼 추가
"use client";

import { useEffect, useRef, useState } from "react";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { markStepCompleteAuto } from "@/utils/stepTracker";
import html2pdf from "html2pdf.js";

export default function ReportPage() {
  const [title, setTitle] = useState("ESG 중요성 평가 보고서");
  const [content, setContent] = useState("");
  const reportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    markStepCompleteAuto();
  }, []);

  const handleGenerate = () => {
    const draft = `1. 평가 개요\n- 이 보고서는 이해관계자 의견 및 전문가 평가를 기반으로 주요 이슈를 도출하고 중요성 매트릭스로 시각화하였습니다.\n\n2. 이슈 선정 결과\n- 총 10개의 이슈가 최종 선정되었으며, 환경: 5개, 사회: 3개, 지배구조: 2개입니다.\n\n3. 중요성 매트릭스\n- 전문가 평가(X축)와 이해관계자 평가(Y축)를 기준으로 이슈들을 사분면에 배치하였습니다.\n\n4. 결론 및 시사점\n- 1사분면에 해당하는 이슈들은 전략 우선 대응이 필요합니다.`;
    setContent(draft);
  };

  const handleDownload = () => {
    if (!reportRef.current) return;
    html2pdf()
      .set({
        margin: 0.5,
        filename: `${title}.pdf`,
        html2canvas: { scale: 2 },
        jsPDF: { unit: "in", format: "a4", orientation: "portrait" },
      })
      .from(reportRef.current)
      .save();
  };

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">보고서 생성</h2>
        <StepBar current="Report" />

        <div className="max-w-6xl mx-auto mt-6 space-y-4">
          <div className="border p-4 rounded bg-gray-50">
            <label className="block text-sm font-medium mb-1">보고서 제목</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full border rounded px-3 py-2"
              placeholder="보고서 제목 입력"
            />
          </div>

          <div className="border p-4 rounded bg-gray-50">
            <label className="block text-sm font-medium mb-1">보고서 내용</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={10}
              className="w-full border rounded px-3 py-2 whitespace-pre-wrap"
              placeholder="보고서 내용을 입력하세요"
            ></textarea>
          </div>

          {/* 버튼 영역 */}
          <div className="flex justify-between">
            <button
              onClick={handleGenerate}
              className="bg-gray-100 text-gray-800 px-6 py-2 rounded hover:bg-gray-200 border"
            >
              보고서 초안 생성
            </button>
            <button
              onClick={handleDownload}
              className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
            >
              보고서 저장하기
            </button>
          </div>

          {/* 숨겨진 PDF 대상 렌더링 */}
          <div className="hidden">
            <div ref={reportRef} className="p-8">
              <h1 className="text-2xl font-bold mb-4">{title}</h1>
              <pre className="whitespace-pre-wrap text-sm text-gray-800">
                {content}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}