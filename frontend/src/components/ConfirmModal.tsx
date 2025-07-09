"use client";

import React from "react";

export default function ConfirmModal({
  title = "확인",
  message = "정말 진행하시겠습니까?",
  groupName = "",
  onConfirm,
  onCancel
}: {
  title?: string;
  message?: string;
  groupName?: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
      <div className="relative w-full max-w-md bg-white rounded-lg shadow-lg border p-6 pointer-events-auto">
        <div className="mb-4">
          <h2 className="text-lg font-bold text-gray-800">{title}</h2>
          {groupName && (
            <p className="text-sm text-blue-600 mt-1">선택 대상: {groupName}</p>
          )}
        </div>

        <p className="text-gray-700 text-sm mb-6">{message}</p>

        <div className="flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm border rounded text-gray-600 hover:bg-gray-100"
          >
            취소
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            확인
          </button>
        </div>
      </div>
    </div>
  );
}
