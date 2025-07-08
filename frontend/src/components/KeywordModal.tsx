// ✅ components/KeywordModal.tsx (다시 생성: 입력창 위에 뜨는 팝업 + 키워드 추가 리스트)
"use client";

import { useState } from "react";

export default function KeywordModal({
  onClose,
  onSave,
}: {
  onClose: () => void;
  onSave: (keyword: string) => void;
}) {
  const [input, setInput] = useState("");
  const [list, setList] = useState<string[]>([]);

  const addKeyword = () => {
    const kw = input.trim();
    if (kw && !list.includes(kw)) {
      setList([...list, kw]);
      setInput("");
    }
  };

  const handleSave = () => {
    list.forEach(onSave);
    onClose();
  };

  return (
    <div className="absolute inset-0 z-30 flex items-center justify-center pointer-events-none">
      <div className="relative w-full max-w-md bg-white rounded-lg p-6 border shadow pointer-events-auto">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-bold">키워드 추가하기</h3>
          <button onClick={onClose} className="text-xl font-bold">×</button>
        </div>

        <div className="bg-gray-100 p-4 rounded-lg border mb-4">
          <label className="block mb-2 font-medium">키워드 입력</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="flex-1 px-4 py-2 border rounded text-gray-900 bg-white"
              placeholder="키워드를 입력하세요"
            />
            <button
              onClick={addKeyword}
              className="border rounded w-10 h-10 text-xl bg-white flex items-center justify-center"
            >
              +
            </button>
          </div>

          <ul className="mt-4 space-y-1 text-sm text-gray-700">
            {list.map((kw, i) => (
              <li key={i} className="bg-white border px-3 py-1 rounded">
                {kw}
              </li>
            ))}
          </ul>
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
