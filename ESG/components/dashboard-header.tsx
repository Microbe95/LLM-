"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { Menu, HelpCircle, LogOut, Bell } from "lucide-react"

interface DashboardHeaderProps {
  user: {
    id: string
    name: string
    email: string
    businessName?: string
  }
}

export default function DashboardHeader({ user }: DashboardHeaderProps) {
  const router = useRouter()

  async function handleLogout() {
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

  return (
    <header className="flex justify-between items-center bg-[#4169e1] text-white px-2 py-1 h-10">
      <div className="flex items-center space-x-4">
        <Menu className="h-5 w-5" />
        <Link href="/dashboard" className="font-bold text-sm">
          Smart ESG
        </Link>
        <div className="flex space-x-2 text-xs">
          <button className="legacy-menu-item bg-blue-700">홈</button>
          <button className="legacy-menu-item">보고서</button>
          <Link href="/settings" className="legacy-menu-item">설정</Link>
          <button className="legacy-menu-item">도움말</button>
        </div>
      </div>
      <div className="flex items-center space-x-2">
        <div className="text-xs mr-2">{user.businessName || user.name}님 환영합니다</div>
        <Bell className="h-4 w-4" />
        <HelpCircle className="h-4 w-4" />
        <button onClick={handleLogout} className="flex items-center">
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </header>
  )
}
