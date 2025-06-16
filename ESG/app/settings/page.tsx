"use client"

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
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
  { icon: Building2, label: "회사 설정", href: "/settings/company", description: "회사 기본 정보를 관리합니다." },
  { icon: Users, label: "사용자 관리", href: "/settings/users", description: "사용자 계정을 관리합니다." },
  { icon: FileText, label: "문서 관리", href: "/settings/documents", description: "ESG 관련 문서를 관리합니다." },
  { icon: Settings, label: "시스템 설정", href: "/settings/system", description: "시스템 환경을 설정합니다." },
  { icon: Bell, label: "알림 설정", href: "/settings/notifications", description: "알림 설정을 관리합니다." },
  { icon: Shield, label: "보안 설정", href: "/settings/security", description: "보안 관련 설정을 관리합니다." },
  { icon: Database, label: "데이터 관리", href: "/settings/data", description: "데이터 백업 및 복구를 관리합니다." },
  { icon: HelpCircle, label: "도움말", href: "/settings/help", description: "도움말 및 가이드를 확인합니다." },
  { icon: Mail, label: "문의하기", href: "/settings/contact", description: "문의사항을 등록합니다." },
]

export default function SettingsPage() {
  const router = useRouter()

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
      } catch (err) {
        console.error('사용자 정보 조회 중 오류:', err)
        router.push('/')
      }
    }

    fetchUserData()
  }, [router])

  return (
    <div className="min-h-[calc(100vh-40px)] flex items-center justify-center bg-gray-100 py-6">
      <div className="bg-white rounded-3xl shadow max-w-3xl w-full mx-4 p-10">
        <h1 className="text-2xl font-bold mb-8 text-center">설정</h1>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {menuItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="flex flex-col items-center justify-center p-6 bg-gray-50 rounded-xl shadow hover:shadow-lg border border-gray-200 hover:border-indigo-400 transition-all group"
            >
              <div className="flex items-center justify-center w-14 h-14 rounded-full bg-indigo-100 mb-3 group-hover:bg-indigo-200">
                <item.icon className="h-7 w-7 text-indigo-600" />
              </div>
              <div className="text-center">
                <h2 className="font-semibold text-gray-900 mb-1">{item.label}</h2>
                <p className="text-sm text-gray-500">{item.description}</p>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
} 