'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle } from 'lucide-react';
import PostcodeSearchModal from './PostcodeSearchModal';

interface CompanyInfo {
  id: string;
  name: string;
  businessNumber: string;
  industry: string;
  address: string;
  postcode: string;
  latitude: number;
  longitude: number;
  unlocode: string;
}

export default function CompanyManagement() {
  const router = useRouter();
  const [companyInfo, setCompanyInfo] = useState<CompanyInfo | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedInfo, setEditedInfo] = useState<CompanyInfo | null>(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isPostcodeModalOpen, setIsPostcodeModalOpen] = useState(false);

  useEffect(() => {
    fetchCompanyInfo();
  }, []);

  const fetchCompanyInfo = async () => {
    try {
      const response = await fetch('/api/company/info', {
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error('사업장 정보를 가져오는데 실패했습니다.');
      }

      const data = await response.json();
      setCompanyInfo(data);
      setEditedInfo(data);
    } catch (err) {
      console.error('사업장 정보 조회 중 오류:', err);
      setError('사업장 정보를 불러오는데 실패했습니다.');
    }
  };

  const handleEdit = () => {
    setIsEditing(true);
  };

  const handleCancel = () => {
    setIsEditing(false);
    setEditedInfo(companyInfo);
    setError('');
  };

  const handleSave = async () => {
    try {
      if (!editedInfo) return;

      const response = await fetch('/api/company/update', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(editedInfo),
      });

      if (!response.ok) {
        throw new Error('사업장 정보 수정에 실패했습니다.');
      }

      setCompanyInfo(editedInfo);
      setIsEditing(false);
      setSuccess('사업장 정보가 성공적으로 수정되었습니다.');
      setTimeout(() => setSuccess(''), 3000);
    } catch (err) {
      console.error('사업장 정보 수정 중 오류:', err);
      setError('사업장 정보 수정에 실패했습니다.');
    }
  };

  const handleAddressSelect = (data: any) => {
    if (!editedInfo) return;
    
    setEditedInfo(prev => {
      if (!prev) return null;
      return {
        ...prev,
        address: data.address,
        postcode: data.zonecode,
        latitude: data.latitude,
        longitude: data.longitude,
        unlocode: data.unlocode || prev.unlocode,
      };
    });
    setIsPostcodeModalOpen(false);
  };

  if (!companyInfo) {
    return <div>로딩 중...</div>;
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">사업장 정보</h2>
      
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      {success && (
        <Alert>
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      <div className="space-y-4">
        <div>
          <Label>사업장 ID</Label>
          <Input
            value={editedInfo?.id || ''}
            disabled
          />
        </div>

        <div>
          <Label>사업장명</Label>
          <Input
            value={editedInfo?.name || ''}
            onChange={(e) => setEditedInfo(prev => prev ? {...prev, name: e.target.value} : null)}
            disabled={!isEditing}
          />
        </div>

        <div>
          <Label>사업자등록번호</Label>
          <Input
            value={editedInfo?.businessNumber || ''}
            onChange={(e) => setEditedInfo(prev => prev ? {...prev, businessNumber: e.target.value} : null)}
            disabled={!isEditing}
          />
        </div>

        <div>
          <Label>업종</Label>
          <Input
            value={editedInfo?.industry || ''}
            onChange={(e) => setEditedInfo(prev => prev ? {...prev, industry: e.target.value} : null)}
            disabled={!isEditing}
          />
        </div>

        <div>
          <Label>주소</Label>
          <div className="flex gap-2">
            <Input
              value={editedInfo?.address || ''}
              disabled
            />
            {isEditing && (
              <Button
                type="button"
                onClick={() => setIsPostcodeModalOpen(true)}
                className="whitespace-nowrap"
              >
                주소 검색
              </Button>
            )}
          </div>
        </div>

        <div>
          <Label>우편번호</Label>
          <Input
            value={editedInfo?.postcode || ''}
            disabled
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label>위도</Label>
            <Input
              value={editedInfo?.latitude || ''}
              disabled
            />
          </div>
          <div>
            <Label>경도</Label>
            <Input
              value={editedInfo?.longitude || ''}
              disabled
            />
          </div>
        </div>

        <div>
          <Label>UNLOCODE</Label>
          <Input
            value={editedInfo?.unlocode || ''}
            disabled
          />
        </div>

        <div className="flex justify-end space-x-2 pt-4">
          {!isEditing ? (
            <Button onClick={handleEdit}>
              수정
            </Button>
          ) : (
            <>
              <Button variant="outline" onClick={handleCancel}>
                취소
              </Button>
              <Button onClick={handleSave}>
                저장
              </Button>
            </>
          )}
        </div>
      </div>

      <PostcodeSearchModal
        isOpen={isPostcodeModalOpen}
        onOpenChange={setIsPostcodeModalOpen}
        onSelect={handleAddressSelect}
      />
    </div>
  );
} 