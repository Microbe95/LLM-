// ✅ Main Page (app/page.tsx) - 너가 제공한 UI 기준 완전 반영
"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ProjectProgressCard } from "@/components/ProjectProgressCard";

export default function MainPage() {
  const [projects, setProjects] = useState<any[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem("projects");
    if (stored) setProjects(JSON.parse(stored));
  }, []);

  return (
    <main className="min-h-screen bg-white p-8">
      {/* ✅ 헤더 */}
      <header className="flex justify-between items-center border-b pb-4 mb-6">
        <div className="flex items-center space-x-2">
          <div className="text-xl font-bold text-gray-800">Auto Mass</div>
          <span className="text-sm text-gray-500">ESG 중대성 평가 대시보드</span>
        </div>
        <Link href="/login">
          <button className="bg-blue-100 text-blue-700 px-4 py-2 rounded-md font-semibold hover:bg-blue-200">
            Login
          </button>
        </Link>
      </header>

      {/* ✅ 본문 영역 */}
      {projects.length > 0 ? (
        <section className="border rounded-lg p-6 shadow-sm">
          <h2 className="text-lg font-semibold mb-1 text-gray-800">현재 프로젝트</h2>
          <p className="text-sm text-blue-600 mb-4">기업별 진단 보기 & 중요성 평가</p>

          {/* ✅ 진행률 바 */}
          <div className="mb-4">
            <div className="text-sm text-gray-600 mb-1">전체 진행률</div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div className="bg-blue-400 h-3 rounded-full" style={{ width: '40%' }}></div>
            </div>
            <div className="text-xs text-right text-gray-500 mt-1">40% 완료</div>
          </div>

          {/* ✅ 평가 단계 바 */}
          <div className="flex items-center justify-between px-4 py-3 bg-gray-50 rounded-md border mb-4">
            {['Issue', 'Survey', 'Evaluate', 'Mapping', 'Report'].map((step, i) => (
              <div key={i} className="text-center">
                <div className="text-blue-600 font-semibold">{step}</div>
                {i < 4 && <div className="text-gray-400">➜</div>}
              </div>
            ))}
          </div>

          {/* ✅ 평가 계속하기 버튼 */}
          <div className="text-right">
            <button className="bg-blue-600 text-white px-4 py-2 rounded-md font-semibold hover:bg-blue-700">
              평가 계속하기
            </button>
          </div>
        </section>
      ) : (
        <div className="text-center border p-10 rounded-lg shadow-sm">
          <p className="mb-6 text-gray-600 text-lg">진행 중인 프로젝트가 없습니다.</p>
          <Link href="/issue/project">
            <button className="bg-blue-700 text-white text-lg px-6 py-3 rounded-md hover:bg-blue-800">
              + 새로운 중요성 평가 시작하기
            </button>
          </Link>
        </div>
      )}
    </main>
  );
}