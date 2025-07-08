// ✅ components/evaluate/IssueTab.tsx
"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const data = [
  { name: "환경 A", value: 80 },
  { name: "환경 B", value: 60 },
  { name: "사회 A", value: 75 },
  { name: "사회 B", value: 45 },
  { name: "지배구조 A", value: 50 },
  { name: "지배구조 B", value: 70 },
];

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
      <div>
        <h3 className="font-semibold mb-2">ESG 이슈별 중요도 점수</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {categories.map((cat, i) => (
          <div key={i} className="bg-gray-50 border rounded p-4 text-sm">
            <h4 className="font-semibold mb-1">{cat.label}</h4>
            <p className="mb-1">평균점수: {cat.평균점수}</p>
            <p className="mb-1 font-medium">Top 5</p>
            <ul className="list-disc list-inside text-gray-700">
              {cat.top5.map((item, j) => (
                <li key={j}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
