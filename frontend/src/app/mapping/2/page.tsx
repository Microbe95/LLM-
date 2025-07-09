// ✅ 개선된 MappingStep2.tsx - 4분면 매트릭스 + 포인트 매핑 구현
"use client";

import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function MappingStep2() {
  const router = useRouter();
  const [xLabel, setXLabel] = useState("전문가 평가");
  const [yLabel, setYLabel] = useState("이해관계자 평가");
  const [areaLabel, setAreaLabel] = useState("중요성 매트릭스");

  const points = [
    { issue: "이슈 A", x: 80, y: 75 },
    { issue: "이슈 B", x: 30, y: 85 },
    { issue: "이슈 C", x: 60, y: 40 },
    { issue: "이슈 D", x: 20, y: 20 },
    { issue: "이슈 E", x: 90, y: 20 },
  ];

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">중요성 매트릭스 매핑</h2>
        <StepBar current="Mapping" />

        <div className="grid grid-cols-5 gap-4 mt-6 max-w-6xl mx-auto">
          {/* 이슈 목록 */}
          <div className="col-span-1 border p-4 rounded bg-gray-50">
            <h3 className="text-sm font-semibold text-gray-600 mb-2">이슈목록</h3>
            <ul className="space-y-2">
              {points.map((p, i) => (
                <li key={i} className="bg-white border p-2 rounded text-sm">
                  {p.issue}
                </li>
              ))}
            </ul>
          </div>

          {/* 매트릭스 영역 */}
          <div className="col-span-4 border rounded p-4 bg-white">
            {/* 축 라벨 입력 */}
            <div className="grid grid-cols-2 gap-2 mb-4">
              <input
                type="text"
                value={xLabel}
                onChange={(e) => setXLabel(e.target.value)}
                placeholder="X축"
                className="border px-3 py-2 rounded text-sm"
              />
              <input
                type="text"
                value={yLabel}
                onChange={(e) => setYLabel(e.target.value)}
                placeholder="Y축"
                className="border px-3 py-2 rounded text-sm"
              />
              <input
                type="text"
                value={areaLabel}
                onChange={(e) => setAreaLabel(e.target.value)}
                placeholder="영역"
                className="border px-3 py-2 rounded text-sm col-span-2"
              />
            </div>

            {/* 매핑 매트릭스 */}
            <div className="relative h-[400px] bg-white border border-gray-300 rounded">
              {/* 축선 */}
              <div className="absolute inset-0 flex">
                <div className="w-1/2 border-r border-gray-300" />
                <div className="w-1/2" />
              </div>
              <div className="absolute inset-0 flex flex-col">
                <div className="h-1/2 border-b border-gray-300" />
                <div className="h-1/2" />
              </div>

              {/* 사분면 라벨 */}
              <div className="absolute top-2 left-2 text-xs text-gray-400">2사분면</div>
              <div className="absolute top-2 right-2 text-xs text-gray-400">1사분면</div>
              <div className="absolute bottom-2 left-2 text-xs text-gray-400">3사분면</div>
              <div className="absolute bottom-2 right-2 text-xs text-gray-400">4사분면</div>

              {/* 포인트 표시 */}
              {points.map((p, i) => (
                <div
                  key={i}
                  className="absolute transform -translate-x-1/2 translate-y-1/2"
                  style={{
                    left: `${p.x}%`,
                    bottom: `${p.y}%`,
                  }}
                >
                  <div className="text-xs bg-blue-600 text-white px-2 py-1 rounded shadow">
                    {p.issue}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex justify-end max-w-6xl mx-auto mt-6">
          <button
            onClick={() => router.push("/report/1")}
            className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
          >
            저장하고 넘어가기
          </button>
        </div>
      </div>
    </main>
  );
}