import React, { useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Detection = {
  authorized: boolean;
  driver_id: string | null;
  driver_name: string;
  confidence: number;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
};

type VerificationResult = {
  detections?: Detection[]; // pode vir ausente em fluxos antigos
  car_id: string;
  authorized_count: number;
  unknown_count: number;
  image_dimensions?: { width: number; height: number };
};

type Props = {
  imageData: string;
  verificationResult: VerificationResult;
  className?: string;
};

export const FaceVerificationVisualizer: React.FC<Props> = ({
  imageData,
  verificationResult,
  className = "",
}) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const draw = () => {
      const cw = img.clientWidth || img.naturalWidth;
      const ch = (img.naturalHeight / img.naturalWidth) * cw;
      canvas.width = cw;
      canvas.height = ch;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, cw, ch);

      // se não houver detecções, apenas exibe a imagem sem overlay
      const dets = Array.isArray(verificationResult?.detections)
        ? verificationResult.detections!
        : [];

      if (dets.length === 0) {
        return;
      }

      const srcW = verificationResult.image_dimensions?.width ?? img.naturalWidth;
      const srcH = verificationResult.image_dimensions?.height ?? img.naturalHeight;
      const scaleX = cw / srcW;
      const scaleY = ch / srcH;

      dets.forEach((d, idx) => {
        const hasBox =
          Number.isFinite(d.x as number) &&
          Number.isFinite(d.y as number) &&
          Number.isFinite(d.width as number) &&
          Number.isFinite(d.height as number);
        if (!hasBox) return;

        const x = (d.x as number) * scaleX;
        const y = (d.y as number) * scaleY;
        const w = (d.width as number) * scaleX;
        const h = (d.height as number) * scaleY;

        const color = d.authorized ? "#22c55e" : "#ef4444";
        ctx.lineWidth = 3;
        ctx.strokeStyle = color;
        ctx.fillStyle = d.authorized ? "rgba(34,197,94,0.18)" : "rgba(239,68,68,0.18)";
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);

        const label = d.driver_name ?? (d.authorized ? "Autorizado" : `Desconhecido #${idx + 1}`);
        const conf = `${(d.confidence ?? 0).toFixed(1)}%`;
        ctx.font = "bold 12px Inter, system-ui, sans-serif";
        const labelW = Math.max(ctx.measureText(label).width, ctx.measureText(conf).width) + 18;
        const labelH = 46;
        const ly = y - labelH - 6 > 0 ? y - 6 : y + h + 6;

        ctx.fillStyle = color;
        ctx.fillRect(x, ly - labelH, labelW, labelH);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x + 8, ly - 26);
        ctx.fillText(conf, x + 8, ly - 10);

        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(x + w - 12, y + 12, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(String(idx + 1), x + w - 12, y + 16);
        ctx.textAlign = "left";
      });
    };

    if (img.complete) draw();
    else img.onload = () => draw();
  }, [imageData, verificationResult]);

  const total = Array.isArray(verificationResult?.detections) ? verificationResult.detections!.length : 0;

  return (
    <Card className={`p-4 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold">Resultado da Verificação</h3>
        <Badge variant={verificationResult.authorized_count > 0 ? "default" : "destructive"}>
          {verificationResult.authorized_count > 0 ? "Autorizado" : "Não autorizado"}
        </Badge>
      </div>
      <div className="relative w-full rounded-lg overflow-hidden border">
        <img ref={imgRef} src={imageData} alt="Imagem verificada" className="w-full h-auto block" />
        <canvas ref={canvasRef} className="absolute top-0 left-0 pointer-events-none" />
      </div>
      {total === 0 && (
        <p className="mt-3 text-sm text-muted-foreground">Nenhuma face detectada.</p>
      )}
    </Card>
  );
};

export default FaceVerificationVisualizer;