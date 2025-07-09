// ✅ components/evaluate/StakeholderTab.tsx 개선본
"use client";

import { useState } from "react";
import ResendingModal from "@/components/evaluate/ResendingModal";

const GROUPS = ["임직원", "고객", "공급업체"];

export default function StakeholderTab() {
  const [selected, setSelected] = useState("임직원");
  const [showModal, setShowModal] = useState(false);

  const infoBox = (label: string, value = "-") => (
    <div className="flex-1 bg-white border rounded p-4">
      <p className="text-sm text-gray-600 mb-1">{label}</p>
      <p className="text-xl font-semibold text-gray-800">{value}</p>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* 이해관계자 선택 + 응답률 진행률 바 */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">이해관계자 그룹 선택</label>
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="border px-3 py-2 rounded w-64"
        >
          {GROUPS.map((g, i) => (
            <option key={i}>{g}</option>
          ))}
        </select>

        <div className="w-full h-4 bg-gray-200 rounded relative">
          <div className="h-full bg-blue-500 rounded" style={{ width: "30%" }}></div>
          <span className="absolute right-2 top-0 text-xs text-white font-semibold leading-4">30%</span>
        </div>
      </div>

      {/* 응답 현황 카드 */}
      <div className="flex gap-4">
        {infoBox("발송 수", "100")}
        {infoBox("응답 수", "30")}
        <div className="flex-1 bg-white border rounded p-4">
          <div className="flex justify-between items-center text-sm text-gray-600 mb-1">
            <span>미응답</span>
            <button
              className="text-blue-600 hover:underline text-xs"
              onClick={() => setShowModal(true)}
            >
              미응답 확인하기
            </button>
          </div>
          <p className="text-xl font-semibold text-gray-800">70</p>
        </div>
      </div>

      {/* 카테고리별 Top5 */}
      <div className="grid grid-cols-3 gap-4 text-sm">
        {["환경", "사회", "지배구조"].map((cat, i) => (
          <div key={i} className="bg-gray-50 border rounded p-4">
            <h4 className="font-semibold mb-1">{cat}</h4>
            <p className="mb-1">평균점수: {60 + i * 5}</p>
            <p className="font-medium mb-1">Top 5</p>
            <ul className="list-disc list-inside text-gray-700">
              {[1, 2, 3, 4, 5].map((n) => (
                <li key={n}>{cat}-지표 {n}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {showModal && <ResendingModal onClose={() => setShowModal(false)} />}
    </div>
  );
}