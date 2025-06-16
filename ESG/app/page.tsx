"use client"

import Link from "next/link"
import { useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { useToast } from '@/components/ui/use-toast'; // Shadcn UI의 useToast

export default function MainPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();

  useEffect(() => {
    if (!searchParams) {
      return;
    }
    const alertType = searchParams.get('alert');
    const message = searchParams.get('message');

    if (alertType === 'login_required' && message) {
      toast({
        title: '알림',
        description: message,
        variant: 'destructive', // 또는 'default' 등 적절한 variant 선택
      });
      // 알림을 띄운 후 URL에서 쿼리 파라미터 제거
      const newPath = window.location.pathname; // 현재 경로 유지
      router.replace(newPath, { scroll: false });
    }
  }, [searchParams, router, toast]);

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <div className="min-h-screen bg-[#f5f6fa]">
        {/* 헤더 */}
        <header className="flex justify-between items-center bg-white border-b border-gray-200 px-6 py-2">
          <nav className="flex space-x-2 text-sm">
            <Link href="/cbam" className="hover:text-blue-600">CBAM 소개</Link>
            <span>|</span>
            <Link href="/guide" className="hover:text-blue-600">이용 안내</Link>
            <span>|</span>
            <Link href="/cbam-calculator" className="hover:text-blue-600">CBAM 계산기</Link>
            <span>|</span>
            <Link href="/mypage" className="hover:text-blue-600">My page</Link>
          </nav>
          <div className="space-x-2">
            <Link href="/" className="px-2 py-1 text-sm border rounded hover:bg-gray-50">Main</Link>
            <Link href="/login" className="px-2 py-1 text-sm border rounded hover:bg-gray-50">Login</Link>
          </div>
        </header>
        {/* 메인 타이틀 */}
        <section className="bg-[#f8f9fa] px-8 py-6">
          <div className="text-lg font-bold mb-1">삼정 KPMG Tech Lap | 3조</div>
          <div className="text-2xl font-bold">CBAM 기준 온실가스 배출량 산출 및 규제 대응 플랫폼 구축 프로젝트</div>
        </section>
        {/* 3가지 블록 */}
        <div className="container mx-auto px-8 py-10 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-[#0a357a] text-white rounded-lg p-6 flex flex-col justify-between min-h-[200px]">
            <div>
              <h3 className="text-xl font-bold mb-2">CBAM 이해하기</h3>
              <p className="text-sm opacity-90">CBAM의 개념을 이해하고,<br/>주요 일정 및 이슈 사항 파악하기</p>
            </div>
            <div className="flex justify-end">
              <Link href="/cbam">
                <button className="bg-white rounded-full w-8 h-8 flex items-center justify-center">
                  <span className="text-[#0a357a]">→</span>
                </button>
              </Link>
            </div>
          </div>
          <div className="bg-[#1656b8] text-white rounded-lg p-6 flex flex-col justify-between min-h-[200px]">
            <div>
              <h3 className="text-xl font-bold mb-2">CBAM 기준<br/>온실가스 배출량 계산하기</h3>
              <p className="text-sm opacity-90">CBAM 기준 온실가스 배출량 계산 및<br/>커뮤니케이션 템플릿 작성하기</p>
            </div>
            <div className="flex justify-end">
              <Link href="/cbam-calculator">
                <button className="bg-white rounded-full w-8 h-8 flex items-center justify-center">
                  <span className="text-[#1656b8]">→</span>
                </button>
              </Link>
            </div>
          </div>
          <div className="bg-[#0070c0] text-white rounded-lg p-6 flex flex-col justify-between min-h-[200px]">
            <div>
              <h3 className="text-xl font-bold mb-2">우리 회사<br/>온실가스 배출량 관리하기</h3>
              <p className="text-sm opacity-90">제품별/공정별 온실가스 배출량 Tracking 및<br/>CBAM 인증서 가격 예측하기</p>
            </div>
            <div className="flex justify-end">
              <Link href="/mypage?tab=emissionStatus">
                <button className="bg-white rounded-full w-8 h-8 flex items-center justify-center">
                  <span className="text-[#0070c0]">→</span>
                </button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </Suspense>
  )
}
