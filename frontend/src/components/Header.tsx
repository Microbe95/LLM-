// ✅ components/Header.tsx (로그인 상태에 따라 로그아웃/로그인 버튼 토글)
"use client";

import Link from "next/link";
import Image from "next/image";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function Header({ showHomeIcon = false }: { showHomeIcon?: boolean }) {
  const [sessionUser, setSessionUser] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    setSessionUser(localStorage.getItem("sessionUser"));
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("sessionUser");
    alert("로그아웃 되었습니다.");
    router.push("/login");
  };

  return (
    <header className="flex justify-between items-center border-b px-8 py-4 bg-white">
      <div className="flex items-center space-x-2">
        <Image src="/icon.png" alt="logo" width={24} height={24} />
        <div className="text-xl font-bold text-gray-800">Auto Mass</div>
        <span className="text-sm text-gray-500">ESG 중대성 평가 대시보드</span>
      </div>
      {showHomeIcon ? (
        <Link href="/">
          <Image src="/home.png" alt="home" width={32} height={32} className="cursor-pointer" />
        </Link>
      ) : sessionUser ? (
        <button
          onClick={handleLogout}
          className="bg-red-100 text-red-600 px-4 py-2 rounded-md font-semibold hover:bg-red-200"
        >
          로그아웃
        </button>
      ) : (
        <Link href="/login">
          <button className="bg-blue-100 text-blue-700 px-4 py-2 rounded-md font-semibold hover:bg-blue-200">
            로그인
          </button>
        </Link>
      )}
    </header>
  );
}