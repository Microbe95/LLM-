"use client";

import { useState, useEffect } from "react";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { useRouter } from "next/navigation";
import { markStepCompleteAuto } from "@/utils/stepTracker";
import { v4 as uuidv4 } from "uuid";

export default function IssueProjectPage() {
  const [division, setDivision] = useState("");
  const [projectName, setProjectName] = useState("");
  const [sessionUser, setSessionUser] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    markStepCompleteAuto();
  }, []);

  useEffect(() => {
    const user = localStorage.getItem("sessionUser");
    if (!user) {
      alert("로그인이 필요합니다.");
      router.push("/login");
    }
    setSessionUser(user);
  }, [router]);

  const handleNext = () => {
    if (!sessionUser) return;
    const trimmedDivision = division.trim();
    const trimmedProjectName = projectName.trim();

    if (!trimmedDivision) {
      alert("분기를 입력해주세요.");
      return;
    }

    const allProjects = JSON.parse(localStorage.getItem("projects") || "{}");
    const userProjects = allProjects[sessionUser] || [];

    // 동일한 분기 중복 방지
    const isDuplicate = userProjects.some(
      (p: any) => p.division === trimmedDivision
    );
    if (isDuplicate) {
      alert("해당 분기의 프로젝트가 이미 존재합니다.");
      return;
    }

    const id = uuidv4();
    const newProject = {
      id,
      division: trimmedDivision,
      name: trimmedProjectName || `${sessionUser} • ${trimmedDivision}`,
      step1: true,
      step2: false,
      step3: false,
      step4: false,
      step5: false,
    };

    const updatedProjects = [newProject, ...userProjects];
    allProjects[sessionUser] = updatedProjects;

    localStorage.setItem("projects", JSON.stringify(allProjects));
    localStorage.setItem("currentProjectId", id);

    router.push("/issue/word");
  };

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">📌 프로젝트 생성</h2>
        <StepBar current="Issue" />

        <div className="bg-gray-100 rounded-lg border p-6 mt-6 max-w-2xl mx-auto shadow-sm">
          <div className="mb-6">
            <label className="block mb-2 font-medium text-gray-700">📆 분기</label>
            <input
              type="text"
              value={division}
              onChange={(e) => setDivision(e.target.value)}
              className="w-full rounded px-4 py-2 border text-gray-900 bg-white"
              placeholder="예: 2025 1분기"
            />
          </div>

          <div className="mb-2">
            <label className="block mb-2 font-medium text-gray-700">📁 프로젝트명</label>
            <input
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className="w-full rounded px-4 py-2 border text-gray-900 bg-white"
              placeholder="입력하지 않으면 자동 생성됩니다."
            />
          </div>

          <p className="text-sm text-gray-500 mt-1">
            프로젝트명 미입력 시 <strong>"사용자ID • 분기"</strong> 형식으로 자동 생성됩니다.
          </p>
        </div>

        <div className="flex justify-end max-w-2xl mx-auto mt-6">
          <button
            onClick={handleNext}
            className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
          >
            저장하고 다음으로 →
          </button>
        </div>
      </div>
    </main>
  );
}
