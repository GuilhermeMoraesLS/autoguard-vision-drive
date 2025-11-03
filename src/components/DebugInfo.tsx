import React from 'react';
import { Card } from '@/components/ui/card';

interface DebugInfoProps {
  showResult: boolean;
  hasVerificationResult: boolean;
  hasCapturedImage: boolean;
  isVerifying: boolean;
  verificationResult?: any;
  capturedImage?: string;
}

export const DebugInfo: React.FC<DebugInfoProps> = ({
  showResult,
  hasVerificationResult,
  hasCapturedImage,
  isVerifying,
  verificationResult,
  capturedImage
}) => {
  return (
    <Card className="p-4 bg-yellow-50 border-yellow-200">
      <h4 className="font-bold text-yellow-800 mb-2">🐛 Debug Info</h4>
      <div className="text-xs space-y-1 text-yellow-700">
        <div>showResult: <strong>{showResult ? 'TRUE' : 'FALSE'}</strong></div>
        <div>hasVerificationResult: <strong>{hasVerificationResult ? 'TRUE' : 'FALSE'}</strong></div>
        <div>hasCapturedImage: <strong>{hasCapturedImage ? 'TRUE' : 'FALSE'}</strong></div>
        <div>isVerifying: <strong>{isVerifying ? 'TRUE' : 'FALSE'}</strong></div>
        {verificationResult && (
          <div>
            <div>Detections: <strong>{verificationResult.detections?.length || 0}</strong></div>
            <div>Authorized: <strong>{verificationResult.authorized_count || 0}</strong></div>
            <div>Unknown: <strong>{verificationResult.unknown_count || 0}</strong></div>
          </div>
        )}
        {capturedImage && (
          <div>Image size: <strong>{Math.round(capturedImage.length / 1024)}KB</strong></div>
        )}
      </div>
    </Card>
  );
};