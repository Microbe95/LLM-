// ✅ components/evaluate/ResendingModal.tsx - 개선된 모달 구현
"use client";

import { useState } from "react";

interface Props {
  onClose: () => void;
}

const dummyList = [
  { name: "홍길동", email: "hong@example.com" },
  { name: "김영희", email: "kim@example.com" },
  { name: "이철수", email: "lee@example.com" },
];

export default function ResendingModal({ onClose }: Props) {
  const [selected, setSelected] = useState<string[]>([]);

  const toggleAll = () => {
    if (selected.length === dummyList.length) setSelected([]);
    else setSelected(dummyList.map((p) => p.email));
  };

  const toggleOne = (email: string) => {
    setSelected((prev) =>
      prev.includes(email) ? prev.filter((e) => e !== email) : [...prev, email]
    );
  };

  const isChecked = (email: string) => selected.includes(email);
  const isAllChecked = selected.length === dummyList.length && selected.length > 0;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg w-full max-w-lg shadow-xl">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">미응답자 확인 및 메일 재발송</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-black text-sm">닫기 ✕</button>
        </div>

        <div className="max-h-64 overflow-y-auto border rounded">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-100 text-gray-700">
              <tr>
                <th className="p-2">
                  <input
                    type="checkbox"
                    checked={isAllChecked}
                    onChange={toggleAll}
                  />
                </th>
                <th className="p-2">이름</th>
                <th className="p-2">이메일</th>
              </tr>
            </thead>
            <tbody>
              {dummyList.map((person, idx) => (
                <tr key={idx} className="border-t">
                  <td className="p-2">
                    <input
                      type="checkbox"
                      checked={isChecked(person.email)}
                      onChange={() => toggleOne(person.email)}
                    />
                  </td>
                  <td className="p-2">{person.name}</td>
                  <td className="p-2">{person.email}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="flex justify-between items-center mt-4 text-sm">
          <p className="text-gray-600">선택된 인원: {selected.length}명</p>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
            >
              취소
            </button>
            <button
              className={`px-4 py-2 rounded text-white ${
                selected.length === 0
                  ? "bg-gray-300 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700"
              }`}
              disabled={selected.length === 0}
            >
              이메일 재발송
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
