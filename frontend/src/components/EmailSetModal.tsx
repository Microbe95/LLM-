// ✅ components/EmailSetModal.tsx (구분 선택 + 이름/이메일 입력 + 엑셀 업로드)
"use client";

import { useState } from "react";

export default function EmailSetModal({ onClose }: { onClose: () => void }) {
  const [type, setType] = useState("이해관계자");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [fileName, setFileName] = useState("");

  const handleExcelUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      // 엑셀 파싱 로직은 추후 추가 예정
      alert(`엑셀 파일 '${file.name}' 업로드됨`);
    }
  };

  return (
    <div className="absolute inset-0 z-40 flex items-center justify-center pointer-events-none">
      <div className="relative bg-white border rounded-lg p-6 w-full max-w-lg shadow pointer-events-auto">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-bold">E-Mail 설정</h3>
          <button onClick={onClose} className="text-xl font-bold">×</button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">구분</label>
            <select
              value={type}
              onChange={(e) => setType(e.target.value)}
              className="w-full border px-4 py-2 rounded bg-white"
            >
              <option>이해관계자</option>
              <option>내부직원</option>
              <option>외부전문가</option>
            </select>
          </div>

          <div className="flex gap-4">
            <div className="flex-1">
              <label className="block text-sm font-medium mb-1">이름</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full border px-4 py-2 rounded"
              />
            </div>
            <div className="flex-1">
              <label className="block text-sm font-medium mb-1">E-Mail</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full border px-4 py-2 rounded"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">엑셀 업로드</label>
            <input
              type="file"
              accept=".xls,.xlsx"
              onChange={handleExcelUpload}
              className="w-full text-sm"
            />
            {fileName && <p className="text-sm text-gray-500 mt-1">📄 {fileName}</p>}
          </div>

          <div className="flex justify-end">
            <button
              onClick={onClose}
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
