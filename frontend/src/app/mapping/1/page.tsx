"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { markStepCompleteAuto } from "@/utils/stepTracker"; // ✅ 자동 진행률 저장 추가

export default function MappingStep1() {
  const router = useRouter();

  // ✅ 진입 시 자동으로 step4 완료 처리
  useEffect(() => {
    markStepCompleteAuto();
  }, []);

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">최종 이슈 선정</h2>
        <StepBar current="Mapping" />

        <div className="grid grid-cols-5 gap-4 mt-6 max-w-6xl mx-auto">
          {/* 평가 결과 테이블 */}
          <div className="col-span-4 border p-4 rounded bg-gray-50">
            <h3 className="text-sm font-semibold text-gray-600 mb-2">ESG 평가 결과</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border text-left">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="p-2">선택</th>
                    <th className="p-2">카테고리</th>
                    <th className="p-2">주제</th>
                    <th className="p-2">topic</th>
                    <th className="p-2">이해관계자 점수</th>
                    <th className="p-2">전문가 점수</th>
                    <th className="p-2">종합 점수</th>
                  </tr>
                </thead>
                <tbody>
                  {[...Array(5)].map((_, i) => (
                    <tr key={i} className="border-t">
                      <td className="p-2"><input type="checkbox" /></td>
                      <td className="p-2">환경</td>
                      <td className="p-2">온실가스</td>
                      <td className="p-2">GHG</td>
                      <td className="p-2">4.2</td>
                      <td className="p-2">4.1</td>
                      <td className="p-2">4.15</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-6">
              <label className="text-sm font-semibold">선정 근거 및 의견</label>
              <textarea className="mt-1 w-full border rounded px-3 py-2 text-sm" rows={4}></textarea>
            </div>
          </div>

          {/* 선정 현황 */}
          <div className="col-span-1 border rounded p-4 bg-white">
            <h4 className="text-sm font-semibold mb-2">선정 현황</h4>
            <p className="text-3xl font-bold text-blue-600 mb-4">10</p>
            <ul className="text-sm text-gray-700">
              <li>환경: 5개</li>
              <li>사회: 3개</li>
              <li>지배구조: 2개</li>
            </ul>
          </div>
        </div>

        <div className="flex justify-end max-w-6xl mx-auto mt-6">
          <button
            onClick={() => router.push("/mapping/2")}
            className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
          >
            저장하고 넘어가기
          </button>
        </div>
      </div>
    </main>
  );
}
