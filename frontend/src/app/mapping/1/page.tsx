// ✅ 개선된 MappingStep1.tsx - 선택 상태 연동 + 시각 강화
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { markStepCompleteAuto } from "@/utils/stepTracker";

const dummyRows = [...Array(5)].map((_, i) => ({
  id: i,
  category: "환경",
  subject: "온실가스",
  topic: "GHG",
  stakeholder: 4.2,
  expert: 4.1,
  total: 4.15,
}));

export default function MappingStep1() {
  const router = useRouter();
  const [selectedIds, setSelectedIds] = useState<number[]>([]);

  useEffect(() => {
    markStepCompleteAuto();
  }, []);

  const toggleCheckbox = (id: number) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((v) => v !== id) : [...prev, id]
    );
  };

  const getCategoryCount = (category: string) => {
    return dummyRows.filter(
      (row) => selectedIds.includes(row.id) && row.category === category
    ).length;
  };

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
                  {dummyRows.map((row) => (
                    <tr
                      key={row.id}
                      className={`border-t ${
                        selectedIds.includes(row.id) ? "bg-blue-50" : ""
                      }`}
                    >
                      <td className="p-2">
                        <input
                          type="checkbox"
                          checked={selectedIds.includes(row.id)}
                          onChange={() => toggleCheckbox(row.id)}
                        />
                      </td>
                      <td className="p-2">{row.category}</td>
                      <td className="p-2">{row.subject}</td>
                      <td className="p-2">{row.topic}</td>
                      <td className="p-2">{row.stakeholder}</td>
                      <td className="p-2">{row.expert}</td>
                      <td className="p-2">{row.total}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-6">
              <label className="text-sm font-semibold">선정 근거 및 의견</label>
              <textarea
                className="mt-1 w-full border rounded px-3 py-2 text-sm"
                rows={4}
                placeholder="선택한 이슈들에 대한 근거 및 의견을 입력해 주세요."
              ></textarea>
            </div>
          </div>

          {/* 선정 현황 */}
          <div className="col-span-1 border rounded p-4 bg-white">
            <h4 className="text-sm font-semibold mb-2">선정 현황</h4>
            <p className="text-3xl font-bold text-blue-600 mb-4">
              {selectedIds.length}
            </p>
            <ul className="text-sm text-gray-700 space-y-1">
              <li className="flex justify-between">
                <span>환경</span>
                <span className="font-medium">{getCategoryCount("환경")}개</span>
              </li>
              <li className="flex justify-between">
                <span>사회</span>
                <span className="font-medium">{getCategoryCount("사회")}개</span>
              </li>
              <li className="flex justify-between">
                <span>지배구조</span>
                <span className="font-medium">{getCategoryCount("지배구조")}개</span>
              </li>
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
