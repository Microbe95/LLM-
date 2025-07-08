// ✅ /app/login/page.tsx (ID 기반 로그인 처리 + 세션 저장)
"use client";

import { useState } from "react";
import Link from "next/link";
import Header from "@/components/Header";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const [id, setId] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  const handleLogin = () => {
    if (!id.trim() || !password.trim()) {
      alert("ID와 비밀번호를 모두 입력해주세요.");
      return;
    }

    const users = JSON.parse(localStorage.getItem("users") || "{}");
    if (!(id in users)) {
      alert("존재하지 않는 ID입니다.");
      return;
    }
    if (users[id].password !== password) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    localStorage.setItem("sessionUser", id);
    alert("로그인 성공: " + id);
    window.location.href = "/";
  };

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="flex items-center justify-center py-12">
        <div className="bg-white border rounded-lg p-8 w-full max-w-sm">
          <label className="block mb-1 text-sm font-medium text-gray-700">ID</label>
          <input
            type="text"
            value={id}
            onChange={(e) => setId(e.target.value)}
            className="w-full border px-4 py-2 rounded mb-4 text-gray-900"
            placeholder="ID를 입력하세요"
          />

          <label className="block mb-1 text-sm font-medium text-gray-700">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full border px-4 py-2 rounded mb-6 text-gray-900"
            placeholder="비밀번호를 입력하세요"
          />

          <button
            onClick={handleLogin}
            className="w-full bg-black text-white py-2 rounded font-semibold hover:bg-gray-900"
          >
            Sign In
          </button>

          <div className="text-center mt-4">
            <Link href="/register" className="text-sm text-blue-600 hover:underline">
              Create an account
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}
