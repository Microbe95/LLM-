"use client";

import { useState } from "react";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import IssueModal from "@/components/IssueModal";
import { useRouter } from "next/navigation";
import SelectedIssueSection from "@/components/SelectedIssueSection"; // 새로 추가됨

export default function IssuePoolPage() {
  const [issues, setIssues] = useState<any[]>([]);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [modalData, setModalData] = useState<any>(null);
  const router = useRouter();

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

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">이슈 풀 구성 관리</h2>
        <StepBar current="Issue" />

        <div className="flex gap-6 mt-6 max-w-6xl mx-auto">
          {/* 1차 이슈풀 */}
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

          {/* 선택된 이슈 섹션 */}
          <div className="w-[320px]">
            <SelectedIssueSection />
          </div>
        </div>

        {/* 좌우 이동 버튼 */}
        <div className="flex justify-between max-w-6xl mx-auto mt-8">
          <button
            onClick={() => router.push("/issue/word")}
            className="bg-gray-200 text-gray-700 px-6 py-2 rounded hover:bg-gray-300"
          >
            ← 이전으로
          </button>

          <button
            onClick={() => router.push("/survey/1")}
            className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
          >
            저장하고 다음으로 →
          </button>
        </div>

        {showModal && (
          <IssueModal
            data={modalData}
            onClose={() => {
              setShowModal(false);
              setEditingIndex(null);
            }}
            onSave={handleSaveIssue}
            onDelete={handleDelete}
          />
        )}
      </div>
    </main>
  );
}