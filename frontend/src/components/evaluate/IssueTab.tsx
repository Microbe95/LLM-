// ✅ components/evaluate/IssueTab.tsx 개선본
"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const data = [
  { name: "환경 A", value: 80 },
  { name: "환경 B", value: 60 },
  { name: "사회 A", value: 75 },
  { name: "사회 B", value: 45 },
  { name: "지배구조 A", value: 50 },
  { name: "지배구조 B", value: 70 },
];

const colorMap: Record<string, string> = {
  환경: "#10B981",
  사회: "#3B82F6",
  지배구조: "#F59E0B",
};

const categories = [
  {
    label: "환경",
    평균점수: 70,
    top5: ["GST-01", "GST-02", "GST-03", "GST-04", "GST-05"],
  },
  {
    label: "사회",
    평균점수: 60,
    top5: ["GST-06", "GST-07", "GST-08", "GST-09", "GST-10"],
  },
  {
    label: "지배구조",
    평균점수: 65,
    top5: ["GST-11", "GST-12", "GST-13", "GST-14", "GST-15"],
  },
];

export default function IssueTab() {
  return (
    <div className="space-y-6">
      {/* 중요도 바 차트 */}
      <div>
        <h3 className="font-semibold mb-2">ESG 이슈별 중요도 점수</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data}>
            <XAxis dataKey="name" />
            <YAxis hide />
            <Tooltip formatter={(v: any) => `${v}점`} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={colorMap[entry.name.split(" ")[0]] || "#8884d8"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* 카테고리별 평균/Top5 */}
      <div className="grid grid-cols-3 gap-4 text-sm">
        {categories.map((cat, i) => (
          <div key={i} className="bg-gray-50 border rounded p-4">
            <h4 className="font-semibold mb-2">{cat.label}</h4>
            <div className="mb-2">
              <p className="text-gray-600">평균점수</p>
              <div className="h-2 bg-gray-200 rounded">
                <div
                  className="h-2 rounded bg-blue-500"
                  style={{ width: `${cat.평균점수}%` }}
                />
              </div>
              <p className="text-right text-xs text-gray-500 mt-1">{cat.평균점수}점</p>
            </div>
            <p className="font-medium mb-1">Top 5</p>
            <ul className="list-decimal list-inside text-gray-700 space-y-0.5">
              {cat.top5.map((item, j) => (
                <li key={j} className="pl-1">{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
