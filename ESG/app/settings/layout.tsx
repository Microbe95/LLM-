"use client"

import type React from "react"
import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { 
  Building2, 
  Users, 
  FileText, 
  Settings, 
  Bell, 
  Shield, 
  Database, 
  HelpCircle, 
  Mail 
} from "lucide-react"

const menuItems = [
  { icon: Building2, label: "회사 설정", href: "/settings/company" },
  { icon: Users, label: "사용자 관리", href: "/settings/users" },
  { icon: FileText, label: "문서 관리", href: "/settings/documents" },
  { icon: Settings, label: "시스템 설정", href: "/settings/system" },
  { icon: Bell, label: "알림 설정", href: "/settings/notifications" },
  { icon: Shield, label: "보안 설정", href: "/settings/security" },
  { icon: Database, label: "데이터 관리", href: "/settings/data" },
  { icon: HelpCircle, label: "도움말", href: "/settings/help" },
  { icon: Mail, label: "문의하기", href: "/settings/contact" },
]

export default function SettingsLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter()
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const response = await fetch('/api/user/me', {
          credentials: 'include'
        })

        if (!response.ok) {
          if (response.status === 401) {
            router.push('/')
            return
          }
          throw new Error('사용자 정보를 가져오는데 실패했습니다.')
        }

        const data = await response.json()
        setUser(data.user)
      } catch (err) {
        console.error('사용자 정보 조회 중 오류:', err)
        router.push('/')
      } finally {
        setLoading(false)
      }
    }

    fetchUserData()
  }, [router])

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">로딩 중...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="flex">
        {/* 좌측 메뉴 */}
        <div className="w-64 bg-white h-screen shadow-md">
          <div className="p-4 border-b">
            <h2 className="text-lg font-semibold text-gray-800">설정</h2>
          </div>
          <nav className="mt-4">
            {menuItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center px-4 py-3 text-gray-700 hover:bg-gray-100 hover:text-indigo-600 transition-colors"
              >
                <item.icon className="w-5 h-5 mr-3" />
                <span>{item.label}</span>
              </Link>
            ))}
          </nav>
        </div>

        {/* 우측 컨텐츠 */}
        <div className="flex-1 p-8">
          <div className="max-w-4xl mx-auto">
            {children}
          </div>
        </div>
      </div>
    </div>
  )
} 