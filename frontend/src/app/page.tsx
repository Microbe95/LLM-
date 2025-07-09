"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Header from "@/components/Header";

const stepRoutes: Record<string, string> = {
  issue: "/issue/project",
  survey: "/survey/1",
  evaluate: "/evaluate/1",
  mapping: "/mapping/1",
  report: "/report/1",
};

const steps = Object.keys(stepRoutes);

export default function MainPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<any[]>([]);
  const [sessionUser, setSessionUser] = useState<string | null>(null);

  useEffect(() => {
    const user = localStorage.getItem("sessionUser");
    setSessionUser(user);

    if (user) {
      const allData = JSON.parse(localStorage.getItem("projects") || "{}");
      setProjects(allData[user] || []);
    }
  }, []);

  const handleContinue = (project: any) => {
    const nextStepIndex = steps.findIndex((_, idx) => !project[`step${idx + 1}`]);
    const safeIndex = nextStepIndex === -1 ? steps.length - 1 : nextStepIndex;
    const nextRoute = stepRoutes[steps[safeIndex]];
    localStorage.setItem("currentProjectId", project.id);
    router.push(nextRoute);
  };

  const handleDelete = (projectId: string) => {
    if (!sessionUser) return;
    if (!confirm("해당 프로젝트를 삭제하시겠습니까?")) return;

    const allData = JSON.parse(localStorage.getItem("projects") || "{}");
    const updated = (allData[sessionUser] || []).filter((p: any) => p.id !== projectId);
    allData[sessionUser] = updated;
    localStorage.setItem("projects", JSON.stringify(allData));
    setProjects(updated);
  };

  const isProjectCompleted = (project: any) =>
    steps.every((_, idx) => project[`step${idx + 1}`] === true);

  return (
    <main className="min-h-screen bg-white p-6">
      <Header />

      {!sessionUser ? (
        <div className="text-center mt-20">
          <p className="text-gray-700 text-lg mb-4">로그인이 필요합니다.</p>
          <Link href="/login">
            <button className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700">
              로그인 하러 가기
            </button>
          </Link>
        </div>
      ) : (
        <div className="flex flex-col items-center">
          {projects.length > 0 && (
            <section className="w-full max-w-3xl bg-white border rounded-xl shadow-md p-6 mt-6">
              <div className="flex justify-between items-center mb-1">
                <div className="text-blue-700 text-sm font-semibold flex items-center gap-2">
                  <span>📁 현재 프로젝트</span>
                  <span className="bg-blue-100 text-blue-700 text-xs px-3 py-1 rounded-full">진행 중</span>
                </div>
              </div>

              <p className="text-gray-800 text-lg font-bold mb-4">기업별 진단 보기 & 중요성 평가</p>

              {projects.map((project, i) => {
                const doneCount = steps.reduce(
                  (count, _, idx) => (project[`step${idx + 1}`] ? count + 1 : count),
                  0
                );
                const progress = Math.round((doneCount / steps.length) * 100);
                const completed = isProjectCompleted(project);

                return (
                  <div key={i} className="relative mb-12 p-4 border rounded-lg bg-white shadow">
                    {/* 프로젝트 정보 */}
                    <p className="text-sm text-gray-500 mb-1">
                      프로젝트명: <strong>{project.name || "이름 없음"}</strong>
                    </p>
                    <p className="text-sm text-gray-500 mb-4">
                      Step {doneCount} of {steps.length}
                    </p>

                    {/* 삭제 버튼 */}
                    <div className="absolute top-4 right-4">
                      <button
                        className="text-sm px-3 py-1 rounded-md bg-red-100 text-red-600 hover:bg-red-200"
                        onClick={() => handleDelete(project.id)}
                      >
                        🗑 삭제하기
                      </button>
                    </div>

                    {/* 진행률 */}
                    <div className="flex justify-between items-center mb-1 text-sm text-gray-600">
                      <span>전체 진행률</span>
                      <span>{progress}% 완료</span>
                    </div>
                    <div className="w-full h-2 bg-gray-200 rounded-full mb-6">
                      <div
                        className="h-2 bg-blue-600 rounded-full transition-all"
                        style={{ width: `${progress}%` }}
                      ></div>
                    </div>

                    {/* 단계 진행 UI */}
                    <div className="flex overflow-x-auto gap-2 mb-6">
                      {steps.map((step, idx) => {
                        const isDone = project[`step${idx + 1}`] === true;
                        const isCurrent =
                          !isDone && steps.slice(0, idx).every((_, j) => project[`step${j + 1}`]);
                        const isPending = !isDone && !isCurrent;

                        let stepClass =
                          "flex items-center gap-1 px-4 py-2 rounded-full text-sm transition whitespace-nowrap min-w-[110px]";

                        if (isDone) {
                          stepClass += " bg-blue-500 text-white hover:bg-blue-600 shadow";
                        } else if (isCurrent) {
                          stepClass +=
                            " border-2 border-blue-600 text-blue-600 font-semibold bg-white hover:bg-blue-50";
                        } else {
                          stepClass += " bg-gray-100 text-gray-400 cursor-not-allowed";
                        }

                        return (
                          <button
                            key={step}
                            onClick={() => {
                              if (!isPending) {
                                localStorage.setItem("currentProjectId", project.id);
                                router.push(stepRoutes[step]);
                              }
                            }}
                            className={stepClass}
                            title={isDone ? "완료됨" : isCurrent ? "다음 단계" : "선행 단계 필요"}
                          >
                            {step.charAt(0).toUpperCase() + step.slice(1)} →
                          </button>
                        );
                      })}
                    </div>

                    {/* 평가 계속하기 버튼 */}
                    {!completed && (
                      <div className="text-right">
                        <button
                          onClick={() => handleContinue(project)}
                          className="bg-blue-700 text-white px-5 py-2 rounded hover:bg-blue-800"
                        >
                          평가 계속하기
                        </button>
                      </div>
                    )}

                    {/* 완료됨 표기 */}
                    {completed && (
                      <div className="text-right">
                        <span className="text-green-600 font-medium">✅ 완료된 프로젝트입니다</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </section>
          )}

          {/* 새로운 프로젝트 CTA */}
          <div className="mt-10">
            <Link href="/issue/project">
              <button className="bg-blue-700 text-white text-md font-medium px-6 py-3 rounded-lg hover:bg-blue-800 shadow">
                + 새로운 중요성 평가 시작하기
              </button>
            </Link>
          </div>
        </div>
      )}
    </main>
  );
}