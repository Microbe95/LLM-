'use client';

import { useState } from 'react';
import CompanyManagement from './company-management';
import UserInfo from './user-info';

const tabs = [
  { id: 'info', label: '회원 정보' },
  { id: 'company', label: '사업장 관리' },
  { id: 'precursors', label: '전구물질 관리' },
  { id: 'ghg', label: '온실가스 배출 현황' },
  { id: 'report', label: '보고서 관리' },
];

export default function MyPageTabs() {
  const [activeTab, setActiveTab] = useState('info'); // 기본으로 회원 정보 탭을 활성화

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`${
                activeTab === tab.id
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
              aria-current={activeTab === tab.id ? 'page' : undefined}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="mt-8">
        {activeTab === 'info' && <UserInfo />}
        {activeTab === 'company' && <CompanyManagement />}
        {activeTab === 'precursors' && (
          <div className="text-center text-gray-500 py-8">
            전구물질 관리 기능은 준비 중입니다.
          </div>
        )}
        {activeTab === 'ghg' && (
          <div className="text-center text-gray-500 py-8">
            온실가스 배출 현황 기능은 준비 중입니다.
          </div>
        )}
        {activeTab === 'report' && (
          <div className="text-center text-gray-500 py-8">
            보고서 관리 기능은 준비 중입니다.
          </div>
        )}
      </div>
    </div>
  );
} 