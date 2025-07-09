"use client";

import { useState, useEffect } from "react";

export default function KeywordModal({
  onClose,
  onUpdate,
  initialKeywords = [],
}: {
  onClose: () => void;
  onUpdate: (keywords: string[]) => void;
  initialKeywords?: string[];
}) {
  const [input, setInput] = useState("");
  const [list, setList] = useState<string[]>([]);

  useEffect(() => {
    setList(initialKeywords);
  }, [initialKeywords]);

  const addKeyword = () => {
    const kw = input.trim();
    if (kw && !list.includes(kw)) {
      setList([...list, kw]);
      setInput("");
    }
  };

  const removeKeyword = (kw: string) => {
    setList(list.filter((k) => k !== kw));
  };

  const handleSave = () => {
    onUpdate(list);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{
        backgroundColor: "rgba(0, 0, 0, 0.3)",
        backdropFilter: "blur(4px)",
        WebkitBackdropFilter: "blur(4px)", // Safari 호환용
      }}
    >
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
              <li
                key={i}
                className="bg-white border px-3 py-1 rounded flex justify-between items-center"
              >
                {kw}
                <button
                  onClick={() => removeKeyword(kw)}
                  className="text-red-600 font-bold ml-2"
                  title="삭제"
                >
                  ×
                </button>
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
