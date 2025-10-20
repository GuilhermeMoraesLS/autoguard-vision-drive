import { useRef, useState, useEffect } from "react";
import { Camera, CameraOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

interface CameraCaptureProps {
  onCapture: (imageData: string) => void;
  isVerifying: boolean;
}

export const CameraCapture = ({ onCapture, isVerifying }: CameraCaptureProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
      }
    } catch (error) {
      console.error("Camera access error:", error);
      toast.error("Não foi possível acessar a câmera");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setIsStreaming(false);
    }
  };

  const captureImage = () => {
    if (!videoRef.current) return;

    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(videoRef.current, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg", 0.8);
    onCapture(imageData);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="w-full">
      <div className="relative aspect-video bg-secondary rounded-lg overflow-hidden border-2 border-border shadow-lg">
        {isStreaming ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            <div className="absolute top-4 right-4 flex items-center gap-2 bg-success/20 backdrop-blur-sm px-3 py-1.5 rounded-full border border-success/50">
              <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
              <span className="text-xs font-medium text-success-foreground">AO VIVO</span>
            </div>
          </>
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center space-y-3">
              <CameraOff className="w-16 h-16 mx-auto text-muted-foreground" />
              <p className="text-muted-foreground text-sm">Câmera desligada</p>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 flex gap-3">
        {!isStreaming ? (
          <Button
            onClick={startCamera}
            className="flex-1 bg-gradient-primary hover:opacity-90 transition-opacity"
            size="lg"
          >
            <Camera className="w-5 h-5 mr-2" />
            Iniciar Câmera
          </Button>
        ) : (
          <>
            <Button
              onClick={stopCamera}
              variant="secondary"
              className="flex-1"
              size="lg"
            >
              <CameraOff className="w-5 h-5 mr-2" />
              Desligar
            </Button>
            <Button
              onClick={captureImage}
              disabled={isVerifying}
              className="flex-1 bg-gradient-primary hover:opacity-90 transition-opacity shadow-glow-primary"
              size="lg"
            >
              {isVerifying ? "Verificando..." : "Verificar Motorista"}
            </Button>
          </>
        )}
      </div>
    </div>
  );
};
