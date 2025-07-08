"use client";

import { useEffect } from "react";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { markStepCompleteAuto } from "@/utils/stepTracker"; // ✅ 자동 저장 로직 추가

export default function ReportPage() {
  // ✅ 진입 시 step5 자동 반영
  useEffect(() => {
    markStepCompleteAuto();
  }, []);

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
              className="w-full border rounded px-3 py-2"
              placeholder="보고서 제목 입력"
            />
          </div>

          <div className="border p-4 rounded bg-gray-50">
            <label className="block text-sm font-medium mb-1">보고서 내용</label>
            <textarea
              rows={10}
              className="w-full border rounded px-3 py-2"
              placeholder="보고서 내용을 입력하세요"
            ></textarea>
          </div>

          <div className="flex justify-end">
            <button className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800">
              보고서 저장하기
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
