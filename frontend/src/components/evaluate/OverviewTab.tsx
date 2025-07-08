// ✅ components/evaluate/OverviewTab.tsx
"use client";

import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";

const COLORS = ["#8884d8", "#82ca9d", "#ffc658"];

const pieData = [
  { name: "환경", value: 40 },
  { name: "사회", value: 30 },
  { name: "지배구조", value: 30 },
];

const barData = [
  { label: "임직원", value: 50 },
  { label: "고객", value: 40 },
  { label: "공급업체", value: 30 },
];

export default function OverviewTab() {
  return (
    <div className="grid grid-cols-2 gap-6">
      <div>
        <h3 className="font-semibold mb-2">이해관계자 그룹별 응답률</h3>
        <ul className="space-y-2">
          {barData.map((item, i) => (
            <li key={i} className="text-sm">
              <div className="flex justify-between mb-1">
                <span>{item.label}</span>
                <span>{item.value}%</span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded">
                <div
                  className="h-2 bg-blue-500 rounded"
                  style={{ width: `${item.value}%` }}
                />
              </div>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h3 className="font-semibold mb-2">ESG 카테고리별 중요도</h3>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={pieData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={60}
              fill="#8884d8"
              label
            >
              {pieData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
