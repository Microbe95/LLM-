"use client";

export default function ProjectProgressCard({ project }: { project: any }) {
  return (
    <section className="border rounded-lg p-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-1 text-gray-800">현재 프로젝트</h2>
      <p className="text-sm text-blue-600 mb-4">기업별 진단 보기 & 중요성 평가</p>

      {/* 진행률 바 */}
      <div className="mb-4">
        <div className="text-sm text-gray-600 mb-1">전체 진행률</div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div className="bg-blue-400 h-3 rounded-full" style={{ width: '40%' }}></div>
        </div>
        <div className="text-xs text-right text-gray-500 mt-1">40% 완료</div>
      </div>

      {/* 평가 단계 */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-50 rounded-md border mb-4">
        {['Issue', 'Survey', 'Evaluate', 'Mapping', 'Report'].map((step, i) => (
          <div key={i} className="text-center">
            <div className="text-blue-600 font-semibold">{step}</div>
            {i < 4 && <div className="text-gray-400">➜</div>}
          </div>
        ))}
      </div>

      <div className="text-right">
        <button className="bg-blue-600 text-white px-4 py-2 rounded-md font-semibold hover:bg-blue-700">
          평가 계속하기
        </button>
      </div>
    </section>
  );
}
