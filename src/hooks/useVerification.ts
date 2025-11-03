import { useState, useCallback } from 'react';
import { toast } from 'sonner';

interface VerificationResponse {
  detections: Array<{
    authorized: boolean;
    driver_id: string | null;
    driver_name: string;
    confidence: number;
    face_index?: number;
    x: number;
    y: number;
    width: number;
    height: number;
  }>;
  car_id: string;
  authorized_count: number;
  unknown_count: number;
  thresholds: {
    strict: number;
    loose: number;
  };
  performance?: {
    faces_processed: number;
    cache_hits: number;
    unique_identifications: number;
  };
  image_dimensions?: {
    width: number;
    height: number;
  };
}

interface UseVerificationProps {
  carId: string;
  authorizedDrivers: Array<{
    id: string;
    name: string;
    photo_url: string;
  }>;
}

export const useVerification = ({ carId, authorizedDrivers }: UseVerificationProps) => {
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<VerificationResponse | null>(null);
  const [capturedImage, setCapturedImage] = useState<string>("");

  const verifyImage = useCallback(async (imageData: string) => {
    if (!authorizedDrivers.length) {
      toast.error("Nenhum motorista autorizado cadastrado");
      return;
    }

    setIsVerifying(true);
    setCapturedImage(imageData);
    
    try {
      const backendBaseUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
      
      const response = await fetch(`${backendBaseUrl}/verify_driver`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageData,
          car_id: carId,
          authorized_drivers: authorizedDrivers
        }),
      });

      if (!response.ok) {
        throw new Error(`Erro na API: ${response.status}`);
      }

      const result = await response.json();
      setVerificationResult(result);
      
      // ✅ Toast melhorado com detalhes das identificações
      if (result.authorized_count > 0) {
        const authorizedNames = result.detections
          .filter((d: any) => d.authorized)
          .map((d: any) => d.driver_name)
          .join(', ');
        
        toast.success(`${result.authorized_count} motorista(s) autorizado(s): ${authorizedNames}`, {
          description: result.unknown_count > 0 
            ? `⚠️ ${result.unknown_count} pessoa(s) desconhecida(s) também detectada(s)`
            : undefined
        });
      } else if (result.unknown_count > 0) {
        toast.error(`${result.unknown_count} pessoa(s) desconhecida(s) detectada(s)`);
      } else {
        toast.warning("Nenhuma face detectada na imagem");
      }
      
    } catch (error) {
      console.error("Erro na verificação:", error);
      toast.error("Erro ao verificar motorista");
      setVerificationResult(null);
    } finally {
      setIsVerifying(false);
    }
  }, [carId, authorizedDrivers]);

  const clearResults = useCallback(() => {
    setVerificationResult(null);
    setCapturedImage("");
  }, []);

  return {
    isVerifying,
    verificationResult,
    capturedImage,
    verifyImage,
    clearResults
  };
};