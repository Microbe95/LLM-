// ✅ components/IssueModal.tsx (업데이트: 배경 불투명 제거하여 위에 띄우기)
"use client";

import { useState, useEffect } from "react";

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
    onSave({ title, desc, source, category });
  };

  return (
    <div className="absolute inset-0 z-30 flex items-center justify-center pointer-events-none">
      <div className="relative w-full max-w-xl bg-white rounded-lg p-6 border shadow pointer-events-auto">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-bold">이슈풀 수정 및 삭제</h3>
          <button onClick={onClose} className="text-xl font-bold">×</button>
        </div>

        <div className="bg-gray-100 p-4 rounded-lg border mb-4">
          <div className="mb-3">
            <label className="block text-sm font-medium mb-1">ESG 이슈명</label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full border px-4 py-2 rounded text-gray-900 bg-white"
            />
          </div>

          <div className="mb-3">
            <label className="block text-sm font-medium mb-1">설명</label>
            <input
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              className="w-full border px-4 py-2 rounded text-gray-900 bg-white"
            />
          </div>

          <div className="mb-3">
            <label className="block text-sm font-medium mb-1">출처</label>
            <input
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="w-full border px-4 py-2 rounded text-gray-900 bg-white"
            />
          </div>

          <div className="mb-3">
            <label className="block text-sm font-medium mb-1">ESG 구분</label>
            <input
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full border px-4 py-2 rounded text-gray-900 bg-white"
            />
          </div>

          {data && (
            <button
              onClick={onDelete}
              className="bg-red-500 text-white px-4 py-2 text-sm rounded hover:bg-red-600 mb-3"
            >
              삭제하기
            </button>
          )}
        </div>

        <div className="flex justify-end">
          <button
            onClick={handleSave}
            className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
          >
            저장하기
          </button>
        </div>
      </div>
    </div>
  );
}