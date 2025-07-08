// ✅ /app/issue/2/page.tsx
"use client";

import { useState } from "react";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import IssueModal from "@/components/IssueModal";
import { useRouter } from "next/navigation"; // 상단에 추가
import { useEffect } from "react";

export default function IssuePoolPage() {
  const [issues, setIssues] = useState<any[]>([]); // [{ title, desc, source, category }]
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [modalData, setModalData] = useState<any>(null);

  const handleAddIssue = () => {
    setModalData(null);
    setShowModal(true);
  };

  const handleEditIssue = (i: number) => {
    setModalData(issues[i]);
    setEditingIndex(i);
    setShowModal(true);
  };

  const handleSaveIssue = (data: any) => {
    const newIssues = [...issues];
    if (editingIndex !== null) newIssues[editingIndex] = data;
    else newIssues.push(data);
    setIssues(newIssues);
    setEditingIndex(null);
    setShowModal(false);
  };

  const handleDelete = () => {
    if (editingIndex !== null) {
      const newIssues = issues.filter((_, i) => i !== editingIndex);
      setIssues(newIssues);
      setEditingIndex(null);
      setShowModal(false);
    }
  };
  const router = useRouter();

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">이슈 풀 구성 관리</h2>
        <StepBar current="Issue" />

        <div className="flex gap-6 mt-6 max-w-6xl mx-auto">
          {/* 1차 이슈풀 영역 */}
          <div className="flex-1 border p-4 rounded-lg bg-gray-50">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-semibold">1차 이슈풀</h3>
              <button
                onClick={handleAddIssue}
                className="text-sm bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700"
              >
                이슈 추가하기
              </button>
            </div>

            {issues.map((issue, i) => (
              <div key={i} className="bg-white border rounded p-3 mb-3">
                <div className="flex justify-between mb-2">
                  <strong>{issue.title}</strong>
                  <button
                    onClick={() => handleEditIssue(i)}
                    className="text-xs text-blue-600 border px-2 py-1 rounded hover:underline"
                  >
                    수정 및 삭제
                  </button>
                </div>
                <p className="text-sm text-gray-600">{issue.desc}</p>
                <p className="text-xs text-gray-400 mt-1">출처: {issue.source}</p>
              </div>
            ))}
          </div>

          {/* 선택된 이슈 영역 (샘플 테이블 구조) */}
          <div className="w-[300px] border p-4 rounded-lg bg-gray-50">
            <h3 className="font-semibold mb-2">선택된 이슈</h3>
            <div className="text-xs text-gray-600">(체크박스/카테고리 선택 구성은 추후 구현)</div>
          </div>
        </div>

        <div className="flex justify-end max-w-6xl mx-auto mt-6">
          <button
  onClick={() => router.push("/survey/1")}
  className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
>
  저장하고 다음으로
</button>
        </div>

        {showModal && (
          <IssueModal
            data={modalData}
            onClose={() => setShowModal(false)}
            onSave={handleSaveIssue}
            onDelete={handleDelete}
          />
        )}
      </div>
    </main>
  );
}