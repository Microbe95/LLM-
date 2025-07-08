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

  // ✅ 자동 저장 (step1 완료)
  useEffect(() => {
    markStepCompleteAuto();
  }, []);

  // ✅ 로그인 확인
  useEffect(() => {
    const user = localStorage.getItem("sessionUser");
    if (!user) {
      alert("로그인이 필요합니다.");
      router.push("/login");
    }
    setSessionUser(user);
  }, [router]);

  // ✅ 프로젝트 저장 및 다음 페이지 이동
  const handleNext = () => {
    if (!sessionUser) return;

    const allProjects = JSON.parse(localStorage.getItem("projects") || "{}");
    const userProjects = allProjects[sessionUser] || [];

    const id = uuidv4(); // 고유 ID 생성

    const newProject = {
      id,
      division,
      name: projectName || `${sessionUser} • ${division}`,
      step1: true,
      step2: false,
      step3: false,
      step4: false,
      step5: false,
    };

    const updatedProjects = [newProject, ...userProjects];
    allProjects[sessionUser] = updatedProjects;

    // ✅ 저장
    localStorage.setItem("projects", JSON.stringify(allProjects));
    localStorage.setItem("currentProjectId", id); // ✅ 현재 프로젝트 ID 저장

    router.push("/issue/word"); // 다음 단계 이동
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
              placeholder="예: 2025 1분기"
            />
          </div>

          <div className="mb-2">
            <label className="block mb-2 font-medium">프로젝트명</label>
            <input
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className="w-full rounded px-4 py-2 border text-gray-900 bg-white"
              placeholder="입력하지 않으면 자동 생성됩니다."
            />
          </div>
          <p className="text-sm text-gray-500">
            프로젝트 명 미기입시 "사용자ID • 분기"로 자동 저장됩니다.
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
