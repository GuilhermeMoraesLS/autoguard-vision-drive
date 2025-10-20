import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { CameraCapture } from "@/components/CameraCapture";
import { DriverStatus } from "@/components/DriverStatus";
import { AccessHistory, AccessRecord } from "@/components/AccessHistory";
import { Shield, ArrowLeft, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const Index = () => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [isVerifying, setIsVerifying] = useState(false);
  const [currentDriver, setCurrentDriver] = useState<{
    authorized: boolean | null;
    name: string;
    timestamp: string;
  }>({
    authorized: null,
    name: "Aguardando...",
    timestamp: "--",
  });
  const [accessHistory, setAccessHistory] = useState<AccessRecord[]>([]);

  useEffect(() => {
    if (!user) {
      navigate("/auth");
    }
  }, [user, navigate]);

  const handleCapture = async (imageData: string) => {
    setIsVerifying(true);
    toast.info("Processando imagem...");

    try {
      // TODO: Substituir por chamada real à API Python
      // const response = await fetch('YOUR_PYTHON_API_URL/verify_driver', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ image: imageData })
      // });
      // const result = await response.json();

      // Simulação temporária (REMOVER quando integrar com Python)
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockResult = Math.random() > 0.5 
        ? { authorized: true, driver: "Guilherme Lopes", timestamp: new Date().toLocaleString("pt-BR") }
        : { authorized: false, driver: "Desconhecido", timestamp: new Date().toLocaleString("pt-BR") };

      // Atualizar status atual
      setCurrentDriver({
        authorized: mockResult.authorized,
        name: mockResult.driver,
        timestamp: mockResult.timestamp,
      });

      // Adicionar ao histórico
      const newRecord: AccessRecord = {
        id: Date.now().toString(),
        driver: mockResult.driver,
        status: mockResult.authorized ? "authorized" : "unauthorized",
        timestamp: mockResult.timestamp,
      };
      setAccessHistory(prev => [newRecord, ...prev]);

      // Feedback
      if (mockResult.authorized) {
        toast.success(`Motorista autorizado: ${mockResult.driver}`);
      } else {
        toast.error("Motorista não reconhecido!");
      }
    } catch (error) {
      console.error("Verification error:", error);
      toast.error("Erro ao verificar motorista");
    } finally {
      setIsVerifying(false);
    }
  };

  if (!user) return null;

  return (
    <div className="min-h-screen bg-gradient-dark">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center shadow-glow-primary">
                <Shield className="w-7 h-7 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">AutoGuard Vision Web</h1>
                <p className="text-sm text-muted-foreground">Verificação de Motorista</p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={() => navigate("/")}
                variant="secondary"
                size="sm"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Voltar
              </Button>
              <Button
                onClick={signOut}
                variant="secondary"
                size="sm"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Sair
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column - Camera */}
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-bold text-foreground mb-2">Captura de Imagem</h2>
              <p className="text-sm text-muted-foreground">
                Posicione o motorista em frente à câmera e clique em "Verificar Motorista"
              </p>
            </div>
            <CameraCapture onCapture={handleCapture} isVerifying={isVerifying} />
          </div>

          {/* Right Column - Status & History */}
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-bold text-foreground mb-2">Status Atual</h2>
              <p className="text-sm text-muted-foreground mb-4">
                Resultado da última verificação
              </p>
              <DriverStatus
                authorized={currentDriver.authorized}
                driverName={currentDriver.name}
                timestamp={currentDriver.timestamp}
              />
            </div>
          </div>
        </div>

        {/* History Table - Full Width */}
        <div className="mt-8">
          <AccessHistory records={accessHistory} />
        </div>

        {/* Integration Note */}
        <div className="mt-8 p-6 bg-accent/10 border border-accent/30 rounded-lg">
          <h3 className="text-lg font-semibold text-foreground mb-2 flex items-center gap-2">
            <Shield className="w-5 h-5 text-accent" />
            Integração com Backend Python
          </h3>
          <p className="text-sm text-muted-foreground mb-3">
            Para conectar ao seu backend Python, edite o arquivo <code className="px-2 py-1 bg-secondary rounded text-accent font-mono text-xs">src/pages/Index.tsx</code> na função <code className="px-2 py-1 bg-secondary rounded text-accent font-mono text-xs">handleCapture</code>:
          </p>
          <pre className="bg-secondary p-4 rounded-lg text-xs text-foreground overflow-x-auto">
{`// Substitua o mock pela chamada real:
const response = await fetch('YOUR_API_URL/verify_driver', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image: imageData })
});
const result = await response.json();`}
          </pre>
        </div>
      </main>
    </div>
  );
};

export default Index;
