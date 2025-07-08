"use client";

import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { useRouter } from "next/navigation";

export default function MappingStep2() {
  const router = useRouter();

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
              {[
                "이슈 A", "이슈 B", "이슈 C", "이슈 D", "이슈 E"
              ].map((issue, i) => (
                <li key={i} className="bg-white border p-2 rounded text-sm">{issue}</li>
              ))}
            </ul>
          </div>

          {/* 매트릭스 영역 */}
          <div className="col-span-4 border rounded p-4 bg-white">
            <div className="grid grid-cols-2 gap-2 mb-4">
              <input type="text" placeholder="X축" className="border px-3 py-2 rounded text-sm" />
              <input type="text" placeholder="Y축" className="border px-3 py-2 rounded text-sm" />
              <input type="text" placeholder="영역" className="border px-3 py-2 rounded text-sm col-span-2" />
            </div>
            <div className="border border-gray-300 rounded h-[400px] bg-gray-50"></div>
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
