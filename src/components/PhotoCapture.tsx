// Crie um novo arquivo: src/components/PhotoCapture.tsx
import { useRef, useState, useCallback } from "react";
import { Camera, CameraOff, RotateCcw, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

interface PhotoCaptureProps {
  onCapture: (photoBlob: Blob) => void;
  onCancel: () => void;
}

export const PhotoCapture = ({ onCapture, onCancel }: PhotoCaptureProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [capturedPhoto, setCapturedPhoto] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: "user", 
          width: { ideal: 640 }, 
          height: { ideal: 480 } 
        },
      });
      
      streamRef.current = stream;
      setIsStreaming(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error("Erro ao acessar câmera:", error);
      toast.error("Erro ao acessar câmera. Verifique as permissões.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsStreaming(false);
    setCapturedPhoto(null);
  };

  const takePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext("2d");

    if (!context) return;

    // Define o tamanho do canvas igual ao vídeo
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Desenha o frame atual do vídeo no canvas
    context.drawImage(video, 0, 0);

    // Converte para data URL
    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    setCapturedPhoto(dataUrl);
  };

  const confirmPhoto = useCallback(() => {
    if (!capturedPhoto) return;

    // Converte data URL para Blob
    fetch(capturedPhoto)
      .then(res => res.blob())
      .then(blob => {
        onCapture(blob);
        stopCamera();
      })
      .catch(error => {
        console.error("Erro ao processar foto:", error);
        toast.error("Erro ao processar foto");
      });
  }, [capturedPhoto, onCapture]);

  const retakePhoto = () => {
    setCapturedPhoto(null);
  };

  return (
    <div className="space-y-4">
      <div className="relative w-full aspect-video bg-secondary rounded-lg overflow-hidden border-2 border-border">
        {capturedPhoto ? (
          <img
            src={capturedPhoto}
            alt="Foto capturada"
            className="w-full h-full object-cover"
          />
        ) : (
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
          />
        )}
        
        {isStreaming && !capturedPhoto && (
          <div className="absolute top-4 left-4 bg-success/20 text-success px-3 py-1 rounded-full text-sm font-medium flex items-center gap-2">
            <div className="w-2 h-2 bg-success rounded-full animate-pulse"></div>
            AO VIVO
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />

      <div className="flex gap-3">
        {!isStreaming ? (
          <>
            <Button
              onClick={startCamera}
              className="flex-1 bg-gradient-primary hover:opacity-90 transition-opacity"
              size="lg"
            >
              <Camera className="w-5 h-5 mr-2" />
              Iniciar Câmera
            </Button>
            <Button
              onClick={onCancel}
              variant="secondary"
              className="flex-1"
              size="lg"
            >
              Cancelar
            </Button>
          </>
        ) : capturedPhoto ? (
          <>
            <Button
              onClick={retakePhoto}
              variant="secondary"
              className="flex-1"
              size="lg"
            >
              <RotateCcw className="w-5 h-5 mr-2" />
              Tirar Novamente
            </Button>
            <Button
              onClick={confirmPhoto}
              className="flex-1 bg-success hover:bg-success/90 transition-colors"
              size="lg"
            >
              <Check className="w-5 h-5 mr-2" />
              Confirmar Foto
            </Button>
          </>
        ) : (
          <>
            <Button
              onClick={stopCamera}
              variant="secondary"
              className="flex-1"
              size="lg"
            >
              <CameraOff className="w-5 h-5 mr-2" />
              Cancelar
            </Button>
            <Button
              onClick={takePhoto}
              className="flex-1 bg-gradient-primary hover:opacity-90 transition-opacity"
              size="lg"
            >
              <Camera className="w-5 h-5 mr-2" />
              Capturar Foto
            </Button>
          </>
        )}
      </div>
    </div>
  );
};