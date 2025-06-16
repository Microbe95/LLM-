"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { LogIn, LogOut } from "lucide-react"

export default function AuthButton() {
  const router = useRouter()
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const checkLoginStatus = async () => {
      try {
        const response = await fetch('/api/user/me', {
          credentials: 'include'
        })
        setIsLoggedIn(response.ok)
      } catch (err) {
        setIsLoggedIn(false)
      } finally {
        setLoading(false)
      }
    }

    checkLoginStatus()
  }, [])

  const handleLogout = async () => {
    try {
      const response = await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include'
      })

      if (!response.ok) {
        throw new Error('로그아웃에 실패했습니다.')
      }

      router.refresh()
      router.push('/')
    } catch (err) {
      console.error('로그아웃 에러:', err)
    }
  }

  const handleLogin = () => {
    router.push('/login')
  }

  if (loading) {
    return null
  }

  return isLoggedIn ? (
    <button
      onClick={handleLogout}
      className="flex items-center text-xs legacy-menu-item"
    >
      <LogOut className="h-4 w-4 mr-1" />
      로그아웃
    </button>
  ) : (
    <button
      onClick={handleLogin}
      className="flex items-center text-xs legacy-menu-item"
    >
      <LogIn className="h-4 w-4 mr-1" />
      로그인
    </button>
  )
} 