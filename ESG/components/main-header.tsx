"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { useState, useEffect } from "react"

export default function MainHeader() {
  const router = useRouter()
  const [isLoggedIn, setIsLoggedIn] = useState(false)

  useEffect(() => {
    const checkAuth = () => {
      const userId = document.cookie.includes('userId')
      setIsLoggedIn(userId)
    }

    checkAuth()
  }, [])

  const handleLogout = async () => {
    try {
      const response = await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
      })

      if (response.ok) {
        setIsLoggedIn(false)
        router.push('/')
      }
    } catch (error) {
      console.error('로그아웃 실패:', error)
    }
  }

  return (
    <header className="flex justify-between items-center bg-white border-b border-gray-200 px-6 py-2">
      <nav className="flex space-x-2 text-sm">
        <Link href="/cbam" className="hover:text-blue-600">CBAM 소개</Link>
        <span>|</span>
        <Link href="/guide" className="hover:text-blue-600">이용 안내</Link>
        <span>|</span>
        <Link href="/cbam-calculator" className="hover:text-blue-600">CBAM 계산기</Link>
        <span>|</span>
        <Link href="/mypage" className="hover:text-blue-600">My Page</Link>
      </nav>

      <div className="space-x-2">
        <Link href="/" className="px-2 py-1 text-sm border rounded hover:bg-gray-50">
          Main
        </Link>
        {isLoggedIn ? (
          <button
            onClick={handleLogout}
            className="px-2 py-1 text-sm border rounded hover:bg-gray-50"
          >
            Logout
          </button>
        ) : (
          <Link
            href="/login"
            className="px-2 py-1 text-sm border rounded hover:bg-gray-50"
          >
            Login
          </Link>
        )}
      </div>
    </header>
  )
}
