"use client";

import { useEffect, useState } from "react";

export default function IssueModal({
  data,
  onClose,
  onSave,
  onDelete,
}: {
  data: any;
  onClose: () => void;
  onSave: (data: any) => void;
  onDelete: () => void;
}) {
  const [title, setTitle] = useState("");
  const [desc, setDesc] = useState("");
  const [source, setSource] = useState("");
  const [category, setCategory] = useState("");

  useEffect(() => {
    if (data) {
      setTitle(data.title || "");
      setDesc(data.desc || "");
      setSource(data.source || "");
      setCategory(data.category || "");
    }
  }, [data]);

  const handleSave = () => {
    if (!title || !category) return alert("이슈명과 ESG 구분은 필수입니다.");
    onSave({ title, desc, source, category });
  };

  return (
  <div className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm">
    <div className="relative w-full max-w-xl bg-white rounded-lg p-6 border shadow-lg animate-fadeIn">

        {/* 헤더 */}
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-bold">이슈 수정/삭제</h3>
          <button onClick={onClose} className="text-xl font-bold">×</button>
        </div>

        {/* 폼 입력 */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-semibold mb-1">ESG 이슈명 *</label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="예: 공급망 리스크"
              className="w-full border px-4 py-2 rounded bg-white text-gray-900"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold mb-1">설명</label>
            <textarea
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              rows={3}
              placeholder="해당 이슈에 대한 간략 설명을 입력하세요."
              className="w-full border px-4 py-2 rounded bg-white text-gray-900"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold mb-1">출처</label>
            <input
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="예: KPMG 보고서, 2024"
              className="w-full border px-4 py-2 rounded bg-white text-gray-900"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold mb-1">ESG 구분 *</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full border px-4 py-2 rounded bg-white text-gray-900"
            >
              <option value="">선택하세요</option>
              <option value="E">환경 (E)</option>
              <option value="S">사회 (S)</option>
              <option value="G">지배구조 (G)</option>
            </select>
          </div>
        </div>

        {/* 버튼 영역 */}
        <div className="flex justify-between items-center mt-6">
          {data && (
            <button
              onClick={onDelete}
              className="text-sm text-red-600 border border-red-500 px-4 py-2 rounded hover:bg-red-50"
            >
              삭제하기
            </button>
          )}
          <div className="flex justify-end flex-1">
            <button
              onClick={handleSave}
              className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
            >
              저장하기
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
