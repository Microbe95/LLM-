// ✅ /app/register/page.tsx (중복 검사 로직 보완 + 가입 후 로그인 이동)
"use client";

import { useState } from "react";
import Header from "@/components/Header";
import { useRouter } from "next/navigation";

export default function RegisterPage() {
  const [id, setId] = useState("");
  const [password, setPassword] = useState("");
  const [isChecked, setIsChecked] = useState(false);
  const router = useRouter();

  const checkDuplicate = () => {
    const users = JSON.parse(localStorage.getItem("users") || "{}");
    if (!id.trim()) {
      alert("ID를 입력해주세요.");
      return;
    }
    if (id in users) {
      alert("이미 존재하는 ID입니다.");
      setIsChecked(false);
    } else {
      alert("사용 가능한 ID입니다.");
      setIsChecked(true);
    }
  };

  const handleRegister = () => {
    if (!id.trim() || !password.trim()) {
      alert("ID와 비밀번호를 모두 입력해주세요.");
      return;
    }
    if (!isChecked) {
      alert("ID 중복 확인이 필요합니다.");
      return;
    }
    const users = JSON.parse(localStorage.getItem("users") || "{}");
    users[id] = { password };
    localStorage.setItem("users", JSON.stringify(users));
    alert("회원가입 완료: " + id);
    setTimeout(() => {
      router.push("/login");
    }, 100);
  };

  return (
    <main className="min-h-screen bg-white text-gray-800">
      <Header showHomeIcon />
      <div className="flex items-center justify-center py-12">
        <div className="bg-white border rounded-lg p-8 w-full max-w-sm">
          <div className="mb-4">
            <div className="flex justify-between mb-1">
              <label className="text-sm font-medium text-gray-700">ID</label>
              <button
                onClick={checkDuplicate}
                className="bg-black text-white px-2 py-1 text-sm rounded hover:bg-gray-800"
              >
                중복확인
              </button>
            </div>
            <input
              type="text"
              value={id}
              onChange={(e) => {
                setId(e.target.value);
                setIsChecked(false); // ID 변경 시 중복 확인 초기화
              }}
              className="w-full border px-4 py-2 rounded text-gray-900"
              placeholder="ID를 입력하세요"
            />
          </div>

          <label className="block mb-1 text-sm font-medium text-gray-700">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full border px-4 py-2 rounded mb-6 text-gray-900"
            placeholder="비밀번호를 입력하세요"
          />

          <button
            onClick={handleRegister}
            className="w-full bg-black text-white py-2 rounded font-semibold hover:bg-gray-900"
          >
            Register
          </button>
        </div>
      </div>
    </main>
  );
}
