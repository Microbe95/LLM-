// ✅ components/evaluate/OverviewTab.tsx 개선본
"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";

const COLORS = ["#10B981", "#3B82F6", "#F59E0B"];

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
      {/* 응답률 바 시각화 */}
      <div>
        <h3 className="font-semibold mb-3">이해관계자 그룹별 응답률</h3>
        <ul className="space-y-3">
          {barData.map((item, i) => (
            <li key={i} className="text-sm">
              <div className="flex justify-between mb-1">
                <span className="text-gray-700 font-medium">{item.label}</span>
                <span className="text-gray-800 font-semibold">{item.value}%</span>
              </div>
              <div className="w-full h-2 bg-gray-200 rounded">
                <div
                  className={`h-2 rounded ${
                    item.value >= 70
                      ? "bg-green-500"
                      : item.value >= 40
                      ? "bg-yellow-400"
                      : "bg-red-400"
                  }`}
                  style={{ width: `${item.value}%` }}
                />
              </div>
            </li>
          ))}
        </ul>
      </div>

      {/* ESG 중요도 원형 차트 */}
      <div>
        <h3 className="font-semibold mb-3">ESG 카테고리별 중요도</h3>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={pieData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={70}
              label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
              labelLine={false}
            >
              {pieData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value: any) => `${value}점`} />
            <Legend verticalAlign="bottom" iconType="circle" />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
