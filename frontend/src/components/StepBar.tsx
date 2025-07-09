"use client";

import React from "react";

interface StepBarProps {
  current: "Issue" | "Survey" | "Evaluate" | "Mapping" | "Report";
}

const steps = ["Issue", "Survey", "Evaluate", "Mapping", "Report"];

const stepLabels: Record<string, string> = {
  Issue: "이슈 설정",
  Survey: "설문 구성",
  Evaluate: "중요도 평가",
  Mapping: "매핑",
  Report: "결과 보고서",
};

export default function StepBar({ current }: StepBarProps) {
  const currentIndex = steps.indexOf(current);

  return (
    <div className="w-full flex flex-wrap items-center justify-center gap-2 sm:gap-4 md:gap-6 mt-4 mb-6">
      {steps.map((step, idx) => {
        const isCurrent = step === current;
        const isDone = idx < currentIndex;
        const isLast = idx === steps.length - 1;

        return (
          <div key={step} className="flex items-center gap-2">
            <div
              className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-semibold border transition
                ${
                  isCurrent
                    ? "bg-blue-600 text-white border-blue-600"
                    : isDone
                    ? "bg-blue-100 text-blue-600 border-blue-200"
                    : "bg-gray-100 text-gray-400 border-gray-300"
                }`}
              title={stepLabels[step]}
            >
              {idx + 1}
            </div>
            <span
              className={`text-sm whitespace-nowrap ${
                isCurrent
                  ? "text-blue-700 font-bold"
                  : isDone
                  ? "text-gray-600"
                  : "text-gray-400"
              }`}
            >
              {stepLabels[step]}
            </span>

            {/* 선 연결 (마지막 step은 제외) */}
            {!isLast && (
              <div
                className={`w-5 sm:w-8 h-0.5 mx-1 transition
                  ${
                    idx < currentIndex
                      ? "bg-blue-500"
                      : "bg-gray-300"
                  }`}
              ></div>
            )}
          </div>
        );
      })}
    </div>
  );
}
