// ✅ /app/issue/project/page.tsx
"use client";

import { useState } from "react";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { useRouter } from "next/navigation";

export default function IssueProjectPage() {
  const [division, setDivision] = useState("");
  const [projectName, setProjectName] = useState("");
  const router = useRouter();

  const handleNext = () => {
    // TODO: 저장 로직 추가 후 이동
    router.push("/issue/word");
  };

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">프로젝트 생성</h2>
        <StepBar current="Issue" />

        <div className="bg-gray-100 rounded-lg border p-6 mt-6 max-w-2xl mx-auto">
          <div className="mb-6">
            <label className="block mb-2 font-medium">분기</label>
            <input
              type="text"
              value={division}
              onChange={(e) => setDivision(e.target.value)}
              className="w-full rounded px-4 py-2 border text-gray-900 bg-white"
            />
          </div>

          <div className="mb-2">
            <label className="block mb-2 font-medium">프로젝트명</label>
            <input
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className="w-full rounded px-4 py-2 border text-gray-900 bg-white"
            />
          </div>
          <p className="text-sm text-gray-500">
            프로젝트 명 미기입시 "기업명"•"분기" 중요성 평가
          </p>
        </div>

        <div className="flex justify-end max-w-2xl mx-auto mt-6">
          <button
            onClick={handleNext}
            className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
          >
            저장하고 다음으로
          </button>
        </div>
      </div>
    </main>
  );
}