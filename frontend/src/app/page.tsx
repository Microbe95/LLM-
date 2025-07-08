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
    const nextRoute = stepRoutes[steps[nextStepIndex]] || "/report/1";
    localStorage.setItem("currentProjectId", project.id); // 현재 프로젝트 ID 저장
    router.push(nextRoute);
  };

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
              <div className="flex justify-between items-center mb-2">
                <p className="text-blue-700 text-sm font-semibold">현재 프로젝트</p>
                <span className="bg-blue-100 text-blue-700 text-xs px-3 py-1 rounded-full">진행 중</span>
              </div>
              <p className="text-gray-700 text-base font-medium mb-2">기업별 진단 보기 & 중요성 평가</p>

              {/* 진행률 */}
              {projects.map((project, i) => {
                const doneCount = steps.reduce(
                  (count, _, idx) => (project[`step${idx + 1}`] ? count + 1 : count),
                  0
                );
                const progress = Math.round((doneCount / steps.length) * 100);

                return (
                  <div key={i}>
                    <div className="flex justify-between items-center mb-1 text-sm text-gray-600">
                      <span>전체 진행률</span>
                      <span>{progress}% 완료</span>
                    </div>
                    <div className="w-full h-2 bg-gray-300 rounded-full mb-6">
                      <div
                        className="h-2 bg-blue-600 rounded-full"
                        style={{ width: `${progress}%` }}
                      ></div>
                    </div>

                    {/* 단계 진행 UI */}
                    <div className="flex justify-between items-center gap-2 mb-6">
                      {steps.map((step, idx) => {
                        const isDone = project[`step${idx + 1}`] === true;
                        const isCurrent =
                          !isDone && steps.slice(0, idx).every((_, j) => project[`step${j + 1}`]);
                        const isPending = !isDone && !isCurrent;

                        let stepClass =
                          "flex items-center gap-1 px-4 py-2 rounded-full text-sm transition whitespace-nowrap";

                        if (isDone) {
                          stepClass +=
                            " bg-blue-500 text-white hover:bg-blue-600 shadow-sm";
                        } else if (isCurrent) {
                          stepClass +=
                            " border-2 border-blue-600 text-blue-600 font-semibold bg-white shadow-sm hover:bg-blue-50";
                        } else {
                          stepClass +=
                            " bg-gray-100 text-gray-400 cursor-not-allowed";
                        }

                        return (
                          <button
                            key={step}
                            onClick={() => {
                              if (!isPending) router.push(stepRoutes[step]);
                            }}
                            className={stepClass}
                          >
                            {step.charAt(0).toUpperCase() + step.slice(1)} →
                          </button>
                        );
                      })}
                    </div>

                    <div className="text-right">
                      <button
                        onClick={() => handleContinue(project)}
                        className="bg-blue-700 text-white px-5 py-2 rounded hover:bg-blue-800"
                      >
                        평가 계속하기
                      </button>
                    </div>
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
