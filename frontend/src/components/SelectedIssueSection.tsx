"use client";

import { useState } from "react";

// 샘플 데이터 (API 연동 또는 props 대체 가능)
const sampleIssues = [
  { title: "온실가스 배출", desc: "탄소배출 관리 필요성", category: "E" },
  { title: "산업재해 예방", desc: "작업장 안전관리 강화", category: "S" },
  { title: "이사회 다양성", desc: "여성 이사 비율 확대", category: "G" },
  { title: "폐기물 처리", desc: "재활용률 제고 방안", category: "E" },
  { title: "지역사회 기여", desc: "사회공헌 활동 지속성", category: "S" },
  { title: "윤리경영", desc: "부패 방지 정책 수립", category: "G" }
];

export default function SelectedIssueSection() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const filteredIssues = selectedCategory
    ? sampleIssues.filter((issue) => issue.category === selectedCategory)
    : [];

  return (
    <div className="w-[320px] border p-4 rounded-lg bg-gray-50">
      <h3 className="font-semibold mb-3 text-center">선택된 이슈</h3>

      {/* 카테고리 선택 */}
      <div className="flex justify-around mb-4">
        {[
          { label: "환경 (E)", value: "E" },
          { label: "사회 (S)", value: "S" },
          { label: "지배구조 (G)", value: "G" }
        ].map((cat) => (
          <button
            key={cat.value}
            onClick={() => setSelectedCategory(cat.value)}
            className={`px-3 py-1 rounded text-sm border font-medium hover:bg-blue-100 transition ${
              selectedCategory === cat.value ? "bg-blue-600 text-white" : "bg-white"
            }`}
          >
            {cat.label}
          </button>
        ))}
      </div>

      {/* 이슈 리스트 */}
      {selectedCategory ? (
        <div className="space-y-3 text-sm">
          {filteredIssues.map((issue, i) => (
            <div
              key={i}
              className="bg-white border rounded p-3 shadow-sm hover:bg-blue-50 cursor-pointer"
            >
              <div className="font-semibold text-gray-800">{issue.title}</div>
              <div className="text-gray-500 text-xs mt-1">{issue.desc}</div>
            </div>
          ))}
          {filteredIssues.length === 0 && (
            <p className="text-gray-400 text-sm text-center">이슈가 없습니다.</p>
          )}
        </div>
      ) : (
        <p className="text-sm text-gray-500 text-center">카테고리를 선택해주세요.</p>
      )}
    </div>
  );
}
