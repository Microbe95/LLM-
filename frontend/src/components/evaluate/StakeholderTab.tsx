// ✅ components/evaluate/StakeholderTab.tsx
"use client";

import { useState } from "react";
import ResendingModal from "@/components/evaluate/ResendingModal";

const GROUPS = ["임직원", "고객", "공급업체"];

export default function StakeholderTab() {
  const [selected, setSelected] = useState("임직원");
  const [showModal, setShowModal] = useState(false);

  const infoBox = (label: string, value = "-") => (
    <div className="flex-1 bg-white border rounded p-3">
      <p className="text-sm text-gray-600 mb-1">{label}</p>
      <div className="h-10 bg-gray-100 rounded"></div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="border px-3 py-2 rounded"
        >
          {GROUPS.map((g, i) => (
            <option key={i}>{g}</option>
          ))}
        </select>

        <div className="w-full h-3 bg-gray-200 rounded">
          <div className="h-3 bg-blue-500 rounded" style={{ width: "30%" }}></div>
        </div>

        <span className="text-sm ml-2">30%</span>
      </div>

      <div className="flex gap-4">
        {infoBox("발송 수")}
        {infoBox("응답 수")}
        <div className="flex-1 bg-white border rounded p-3">
          <p className="text-sm text-gray-600 mb-1 flex justify-between items-center">
            미응답
            <button
              className="text-xs text-blue-600 hover:underline"
              onClick={() => setShowModal(true)}
            >
              미응답 확인하기
            </button>
          </p>
          <div className="h-10 bg-gray-100 rounded"></div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 text-sm">
        {[
          { label: "환경", avg: 70, top: ["GST-01", "GST-02"] },
          { label: "사회", avg: 65, top: ["GST-03", "GST-04"] },
          { label: "지배구조", avg: 68, top: ["GST-05", "GST-06"] },
        ].map((cat, i) => (
          <div key={i} className="bg-gray-50 border rounded p-4">
            <h4 className="font-semibold mb-1">{cat.label}</h4>
            <p className="mb-1">평균점수: {cat.avg}</p>
            <p className="font-medium">Top 5</p>
            <ul className="list-disc list-inside">
              {cat.top.map((item, j) => (
                <li key={j}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {showModal && <ResendingModal onClose={() => setShowModal(false)} />}
    </div>
  );
}
