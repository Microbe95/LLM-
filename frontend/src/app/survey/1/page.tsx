"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { markStepCompleteAuto } from "@/utils/stepTracker";

export default function SurveyPage1() {
  const router = useRouter();
  const [sessionUser, setSessionUser] = useState<string | null>(null);
  const [date, setDate] = useState(() => new Date().toISOString().split("T")[0]);
  const [time, setTime] = useState(() => new Date().toTimeString().slice(0, 5));
  const [title, setTitle] = useState("");
  const [group, setGroup] = useState("");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");

  // ✅ step2 자동 완료
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

  // ✅ 저장 + 다음 이동 + step2 true 처리
  const handleSaveAndNext = () => {
    if (!sessionUser) return;

    // 1. 설문 저장
    const allData = JSON.parse(localStorage.getItem("surveyData") || "{}");
    const userData = allData[sessionUser] || {};
    userData["step1"] = { date, time, title, group, fromDate, toDate };
    allData[sessionUser] = userData;
    localStorage.setItem("surveyData", JSON.stringify(allData));

    // 2. 진행률 step2 업데이트
    const projects = JSON.parse(localStorage.getItem("projects") || "{}");
    const currentId = localStorage.getItem("currentProjectId");
    const userProjects = projects[sessionUser] || [];

    const updated = userProjects.map((p: any) =>
      p.id === currentId ? { ...p, step2: true } : p
    );

    projects[sessionUser] = updated;
    localStorage.setItem("projects", JSON.stringify(projects));

    // 3. 이동
    router.push("/evaluate/1");
  };

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">설문조사 설정</h2>
        <StepBar current="Survey" />

        <div className="grid grid-cols-5 gap-4 mt-6 max-w-5xl mx-auto">
          <button
            onClick={() => router.back()}
            className="col-span-1 bg-blue-500 text-white px-4 py-2 rounded"
          >
            ←
          </button>

          <div className="col-span-4 flex gap-4 items-center">
            <label className="font-semibold">DATE</label>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="border px-3 py-2 rounded"
            />
            <label className="font-semibold">TIME</label>
            <input
              type="time"
              value={time}
              onChange={(e) => setTime(e.target.value)}
              className="border px-3 py-2 rounded"
            />
          </div>

          <div className="col-span-2 flex flex-col gap-3 bg-gray-50 border rounded p-4">
            <label className="text-sm font-medium">설문 제목</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="border px-3 py-2 rounded"
            />

            <label className="text-sm font-medium">이해관계자 구분</label>
            <input
              type="text"
              value={group}
              onChange={(e) => setGroup(e.target.value)}
              className="border px-3 py-2 rounded"
            />

            <label className="text-sm font-medium">설문 기간</label>
            <div className="flex gap-2 items-center">
              <input
                type="date"
                value={fromDate}
                onChange={(e) => setFromDate(e.target.value)}
                className="border px-3 py-2 rounded"
              />
              <span className="text-gray-500">~</span>
              <input
                type="date"
                value={toDate}
                onChange={(e) => setToDate(e.target.value)}
                className="border px-3 py-2 rounded"
              />
            </div>

            <button
              onClick={() => router.push("/survey/main")}
              className="mt-4 bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
            >
              설문 생성하러 가기 →
            </button>
          </div>

          <div className="col-span-3 border bg-gray-50 rounded p-4">
            <h3 className="font-semibold mb-2">설문 현황</h3>
            <div className="h-40 bg-white border rounded"></div>
          </div>
        </div>

        <div className="flex justify-end max-w-5xl mx-auto mt-6">
          <button
            onClick={handleSaveAndNext}
            className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
          >
            저장하고 넘어가기
          </button>
        </div>
      </div>
    </main>
  );
}
