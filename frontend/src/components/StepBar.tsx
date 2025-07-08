// ✅ components/StepBar.tsx
"use client";

import React from "react";

const steps = ["Issue", "Survey", "Evaluate", "Mapping", "Report"];

export default function StepBar({ current }: { current: string }) {
  return (
    <div className="flex items-center justify-center space-x-4 px-4 py-3 border-b border-gray-200 bg-white">
      {steps.map((step, i) => {
        const isActive = step.toLowerCase() === current.toLowerCase();
        return (
          <div key={i} className="flex items-center space-x-1">
            <button
              className={`px-3 py-1 rounded text-sm font-medium border transition-all duration-150 ${
                isActive
                  ? "bg-blue-700 text-white border-blue-700"
                  : "bg-blue-100 text-blue-700 border-blue-300"
              }`}
              disabled
            >
              {step}
            </button>
            {i < steps.length - 1 && <span className="text-blue-600 font-bold">➤</span>}
          </div>
        );
      })}
    </div>
  );
}
