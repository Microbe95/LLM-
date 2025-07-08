// ✅ /app/issue/word/page.tsx
"use client";

import { useState } from "react";
import Header from "@/components/Header";
import StepBar from "@/components/StepBar";
import { useRouter } from "next/navigation";
import KeywordModal from "@/components/KeywordModal";

export default function IssueWordPage() {
  const [company, setCompany] = useState("");
  const [industry, setIndustry] = useState("");
  const [location, setLocation] = useState("");
  const [keywords, setKeywords] = useState<string[]>([]);
  const [showModal, setShowModal] = useState(false);
  const router = useRouter();

  const handleAddKeyword = (newKeyword: string) => {
    setKeywords((prev) => [...prev, newKeyword]);
  };

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="px-6 py-4">
        <h2 className="text-xl font-bold mb-2">이슈 풀 구성 관리</h2>
        <StepBar current="Issue" />

        <div className="bg-gray-100 rounded-lg border p-6 mt-6 max-w-2xl mx-auto">
          <div className="mb-4">
            <label className="block mb-1 font-medium">기업명</label>
            <input
              type="text"
              value={company}
              onChange={(e) => setCompany(e.target.value)}
              className="w-full rounded px-4 py-2 border text-gray-900 bg-white"
            />
          </div>
          <div className="mb-4">
            <label className="block mb-1 font-medium">산업군</label>
            <input
              type="text"
              value={industry}
              onChange={(e) => setIndustry(e.target.value)}
              className="w-full rounded px-4 py-2 border text-gray-900 bg-white"
            />
          </div>
          <div className="mb-4">
            <label className="block mb-1 font-medium">소재지</label>
            <input
              type="text"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              className="w-full rounded px-4 py-2 border text-gray-900 bg-white"
            />
          </div>

          <div>
            <label className="block mb-1 font-medium">키워드</label>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowModal(true)}
                className="text-xl border rounded w-8 h-8 flex items-center justify-center bg-white"
              >
                +
              </button>
              <span className="text-sm text-gray-600">{keywords.join(", ")}</span>
            </div>
          </div>
        </div>

        <div className="flex justify-end max-w-2xl mx-auto mt-6">
          <button
            onClick={() => router.push("/issue/2")}
            className="bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800"
          >
            저장하고 다음으로
          </button>
        </div>

        {showModal && (
          <KeywordModal
            onClose={() => setShowModal(false)}
            onSave={(kw) => {
              handleAddKeyword(kw);
              setShowModal(false);
            }}
          />
        )}
      </div>
    </main>
  );
}
