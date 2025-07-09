"use client";

import { useState } from "react";

interface EmailEntry {
  name: string;
  email: string;
}

export default function EmailSetModal({ onClose }: { onClose: () => void }) {
  const [type, setType] = useState("이해관계자");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [fileName, setFileName] = useState("");
  const [emailList, setEmailList] = useState<EmailEntry[]>([]);

  const handleAddEntry = () => {
    if (name && email) {
      setEmailList([...emailList, { name, email }]);
      setName("");
      setEmail("");
    }
  };

  const handleExcelUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      // 엑셀 파싱 로직은 추후 연결
      alert(`엑셀 파일 '${file.name}' 업로드됨 (미리보기 기능 예정)`);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm">
      <div className="bg-white w-full max-w-2xl p-6 rounded shadow-lg">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-bold">📧 Email 설정</h2>
          <button onClick={onClose} className="text-xl font-bold">×</button>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-1">구분</label>
            <select
              value={type}
              onChange={(e) => setType(e.target.value)}
              className="w-full border px-3 py-2 rounded"
            >
              <option>이해관계자</option>
              <option>내부직원</option>
              <option>외부전문가</option>
            </select>
          </div>

          <div className="col-span-2 flex gap-3">
            <div className="flex-1">
              <label className="block text-sm font-medium mb-1">이름</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full border px-3 py-2 rounded"
              />
            </div>
            <div className="flex-1">
              <label className="block text-sm font-medium mb-1">E-Mail</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full border px-3 py-2 rounded"
              />
            </div>
            <button
              onClick={handleAddEntry}
              className="self-end bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              추가
            </button>
          </div>
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">엑셀 업로드</label>
          <input
            type="file"
            accept=".xls,.xlsx"
            onChange={handleExcelUpload}
            className="text-sm"
          />
          {fileName && <p className="text-sm text-gray-500 mt-1">📄 {fileName}</p>}
        </div>

        {emailList.length > 0 && (
          <div className="mb-4">
            <h4 className="font-medium text-sm mb-2">등록된 목록</h4>
            <ul className="border rounded text-sm max-h-40 overflow-auto">
              {emailList.map((entry, idx) => (
                <li key={idx} className="flex justify-between px-3 py-1 border-b last:border-b-0">
                  <span>{entry.name}</span>
                  <span>{entry.email}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

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
  );
}
