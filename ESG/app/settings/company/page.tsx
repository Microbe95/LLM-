"use client"

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'

interface CompanyInfo {
  id: string
  businessName: string
  businessNameEng: string
  address: string
  industry: string
  postalCode: string
  postBox: string
  city: string
  country: string
  unlocode: string
  latitude: number
  longitude: number
  representativeName: string
  phoneNumber: string
}

export default function CompanySettingsPage() {
  const router = useRouter()
  const [companyInfo, setCompanyInfo] = useState<CompanyInfo | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  useEffect(() => {
    fetchCompanyInfo()
  }, [])

  const fetchCompanyInfo = async () => {
    try {
      const response = await fetch('/api/user/me', {
        credentials: 'include'
      })

      if (!response.ok) {
        throw new Error('회사 정보를 가져오는데 실패했습니다.')
      }

      const data = await response.json()
      setCompanyInfo(data.user)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setError('')
    setSuccess('')
    setIsSaving(true)

    try {
      const response = await fetch('/api/settings/company', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(companyInfo),
        credentials: 'include'
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || '회사 정보 저장에 실패했습니다.')
      }

      setSuccess('회사 정보가 성공적으로 저장되었습니다.')
    } catch (err: any) {
      setError(err.message)
    } finally {
      setIsSaving(false)
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setCompanyInfo(prev => {
      if (!prev) return prev
      return {
        ...prev,
        [name]: name === 'latitude' || name === 'longitude' ? parseFloat(value) : value
      }
    })
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">로딩 중...</p>
        </div>
      </div>
    )
  }

  if (!companyInfo) {
    return (
      <div className="bg-white p-8 rounded-lg shadow-md">
        <div className="text-red-600">
          <p>회사 정보를 불러올 수 없습니다.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white p-8 rounded-lg shadow-md">
      <h1 className="text-2xl font-bold mb-6">회사 설정</h1>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 text-red-600 rounded">
          {error}
        </div>
      )}

      {success && (
        <div className="mb-4 p-4 bg-green-50 border border-green-200 text-green-600 rounded">
          {success}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              회사명 (한글)
            </label>
            <input
              type="text"
              name="businessName"
              value={companyInfo.businessName}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              회사명 (영문)
            </label>
            <input
              type="text"
              name="businessNameEng"
              value={companyInfo.businessNameEng}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              대표자명
            </label>
            <input
              type="text"
              name="representativeName"
              value={companyInfo.representativeName}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              전화번호
            </label>
            <input
              type="tel"
              name="phoneNumber"
              value={companyInfo.phoneNumber}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              업종
            </label>
            <input
              type="text"
              name="industry"
              value={companyInfo.industry}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              우편번호
            </label>
            <input
              type="text"
              name="postalCode"
              value={companyInfo.postalCode}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              주소
            </label>
            <input
              type="text"
              name="address"
              value={companyInfo.address}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              도시
            </label>
            <input
              type="text"
              name="city"
              value={companyInfo.city}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              국가
            </label>
            <input
              type="text"
              name="country"
              value={companyInfo.country}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              UN/LOCODE
            </label>
            <input
              type="text"
              name="unlocode"
              value={companyInfo.unlocode}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              사서함
            </label>
            <input
              type="text"
              name="postBox"
              value={companyInfo.postBox}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              위도
            </label>
            <input
              type="number"
              step="any"
              name="latitude"
              value={companyInfo.latitude}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              경도
            </label>
            <input
              type="number"
              step="any"
              name="longitude"
              value={companyInfo.longitude}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              required
            />
          </div>
        </div>

        <div className="flex justify-end space-x-4">
          <button
            type="button"
            onClick={() => router.back()}
            className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            취소
          </button>
          <button
            type="submit"
            disabled={isSaving}
            className={`px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
              isSaving ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            {isSaving ? '저장 중...' : '저장'}
          </button>
        </div>
      </form>
    </div>
  )
} 