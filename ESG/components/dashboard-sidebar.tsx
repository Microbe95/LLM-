"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  Home,
  PieChart,
  Settings,
  Users,
  Leaf,
  Building2,
  Scale,
  FileBarChart,
  MessageSquare,
  HelpCircle,
} from "lucide-react"

export default function DashboardSidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-[200px] bg-white border-r border-gray-300 min-h-[calc(100vh-40px)]">
      <div className="p-2">
        <div className="grid grid-cols-3 gap-1">
          <Link href="/dashboard" className="menu-icon">
            <div className="menu-icon-img">
              <Home className="h-6 w-6 text-blue-600" />
            </div>
            <span className="text-xs">홈</span>
          </Link>

          <Link href="/dashboard/environment" className="menu-icon">
            <div className="menu-icon-img">
              <Leaf className="h-6 w-6 text-green-600" />
            </div>
            <span className="text-xs">환경</span>
          </Link>

          <Link href="/dashboard/social" className="menu-icon">
            <div className="menu-icon-img">
              <Users className="h-6 w-6 text-orange-600" />
            </div>
            <span className="text-xs">사회</span>
          </Link>

          <Link href="/dashboard/governance" className="menu-icon">
            <div className="menu-icon-img">
              <Scale className="h-6 w-6 text-purple-600" />
            </div>
            <span className="text-xs">지배구조</span>
          </Link>

          <Link href="/dashboard/reports" className="menu-icon">
            <div className="menu-icon-img">
              <FileBarChart className="h-6 w-6 text-blue-600" />
            </div>
            <span className="text-xs">보고서</span>
          </Link>

          <Link href="/dashboard/statistics" className="menu-icon">
            <div className="menu-icon-img">
              <PieChart className="h-6 w-6 text-red-600" />
            </div>
            <span className="text-xs">통계</span>
          </Link>

          <Link href="/dashboard/clients" className="menu-icon">
            <div className="menu-icon-img">
              <Building2 className="h-6 w-6 text-gray-600" />
            </div>
            <span className="text-xs">거래처</span>
          </Link>

          <Link href="/settings" className="menu-icon">
            <div className="menu-icon-img">
              <Settings className="h-6 w-6 text-gray-600" />
            </div>
            <span className="text-xs">설정</span>
          </Link>

          <Link href="/dashboard/help" className="menu-icon">
            <div className="menu-icon-img">
              <HelpCircle className="h-6 w-6 text-blue-600" />
            </div>
            <span className="text-xs">도움말</span>
          </Link>
        </div>
      </div>

      <div className="mt-4 border-t border-gray-300 pt-2">
        <div className="px-3 py-1 text-sm font-bold bg-gray-100">공지사항</div>
        <div className="p-2 text-xs">
          <p className="mb-1">· ESG 보고서 제출 기한: 2023-12-31</p>
          <p className="mb-1">· 탄소배출량 측정 가이드라인 업데이트</p>
          <p className="mb-1">· 신규 ESG 평가 지표 안내</p>
        </div>
      </div>

      <div className="mt-2 border-t border-gray-300 pt-2">
        <div className="px-3 py-1 text-sm font-bold bg-gray-100">고객센터</div>
        <div className="p-2 text-xs flex justify-between">
          <div className="flex items-center">
            <MessageSquare className="h-3 w-3 mr-1" />
            <span>문의하기</span>
          </div>
          <div>1688-0000</div>
        </div>
      </div>
    </aside>
  )
}
