import { CheckCircle2, XCircle, Clock } from "lucide-react";
import { Card } from "@/components/ui/card";

interface DriverStatusProps {
  authorized: boolean | null;
  driverName: string;
  timestamp: string;
}

export const DriverStatus = ({ authorized, driverName, timestamp }: DriverStatusProps) => {
  if (authorized === null) {
    return (
      <Card className="p-6 bg-card border-border">
        <div className="text-center space-y-4">
          <div className="w-20 h-20 mx-auto rounded-full bg-secondary/50 flex items-center justify-center">
            <Clock className="w-10 h-10 text-muted-foreground" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">Aguardando Verificação</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Capture uma imagem para identificar o motorista
            </p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className={`p-6 border-2 transition-all duration-300 ${
      authorized 
        ? "bg-success/5 border-success shadow-glow-success" 
        : "bg-danger/5 border-danger shadow-glow-danger"
    }`}>
      <div className="flex items-start gap-4">
        <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
          authorized ? "bg-success/20" : "bg-danger/20"
        }`}>
          {authorized ? (
            <CheckCircle2 className="w-8 h-8 text-success" />
          ) : (
            <XCircle className="w-8 h-8 text-danger" />
          )}
        </div>
        
        <div className="flex-1 space-y-2">
          <div className="flex items-center gap-2">
            <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
              authorized 
                ? "bg-success text-success-foreground" 
                : "bg-danger text-danger-foreground"
            }`}>
              {authorized ? "AUTORIZADO" : "DESCONHECIDO"}
            </div>
          </div>
          
          <div>
            <h3 className="text-xl font-bold text-foreground">{driverName}</h3>
            <div className="flex items-center gap-2 mt-1 text-sm text-muted-foreground">
              <Clock className="w-4 h-4" />
              <span>{timestamp}</span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};
