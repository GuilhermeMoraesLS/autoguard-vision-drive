import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { CameraCapture } from "@/components/CameraCapture";
import { DriverStatus } from "@/components/DriverStatus";
import { AccessHistory, AccessRecord } from "@/components/AccessHistory";
import { Shield, ArrowLeft, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

interface AuthorizedDriver {
  id: string;
  name: string;
  photo_url: string;
}

interface Car {
  id: string;
  brand: string;
  model: string;
  plate: string;
}

const Index = () => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const { id: carId } = useParams<{ id: string }>();
  const [car, setCar] = useState<Car | null>(null);
  const [authorizedDrivers, setAuthorizedDrivers] = useState<AuthorizedDriver[]>([]);
  const [isLoadingCar, setIsLoadingCar] = useState(true);
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
      return;
    }
    if (!carId) {
      toast.error("Carro não informado");
      navigate("/");
      return;
    }
    const fetchData = async () => {
      try {
        const { data: carData, error: carError } = await supabase
          .from("cars")
          .select("id, brand, model, plate, user_id")
          .eq("id", carId)
          .single();

        if (carError || !carData) {
          toast.error("Carro não encontrado");
          navigate("/");
          return;
        }

        if (carData.user_id !== user.id) {
          toast.error("Você não tem acesso a este veículo");
          navigate("/");
          return;
        }

        setCar({
          id: carData.id,
          brand: carData.brand,
          model: carData.model,
          plate: carData.plate,
        });

        const { data: driversData, error: driversError } = await supabase
          .from("authorized_drivers")
          .select("id, name, photo_url")
          .eq("car_id", carId)
          .order("created_at", { ascending: false });

        if (driversError) {
          toast.error("Erro ao carregar motoristas autorizados");
          setAuthorizedDrivers([]);
        } else {
          setAuthorizedDrivers(driversData || []);
        }
      } catch {
        toast.error("Erro ao carregar dados do veículo");
        navigate("/");
      } finally {
        setIsLoadingCar(false);
      }
    };

    fetchData();
  }, [user, carId, navigate]);

  const handleCapture = async (imageData: string) => {
    if (!carId) {
      toast.error("Carro não informado");
      return;
    }

    if (!authorizedDrivers.length) {
      toast.error("Nenhum motorista autorizado cadastrado para este veículo");
      return;
    }

    setIsVerifying(true);
    toast.info("Processando imagem...");
    setCurrentDriver({ authorized: null, name: "Verificando...", timestamp: "--" });

    try {
      // --- CHAMADA PARA O BACKEND PYTHON ---
      const backendBaseUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

      const response = await fetch(`${backendBaseUrl}/verify_driver`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageData,
          car_id: carId,
          authorized_drivers: authorizedDrivers.map((driver) => ({
            id: driver.id,
            name: driver.name,
            photo_url: driver.photo_url,
          })),
        }),
      });
      // --- FIM DA CHAMADA ---

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: `Erro na API: ${response.status} ${response.statusText}` }));
        throw new Error(errorData.error || `Erro na API: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      const recognizedName = result.driver_name ?? "Desconhecido";
      const timestamp = new Date().toLocaleString("pt-BR");

      setCurrentDriver({
        authorized: result.authorized,
        name: recognizedName,
        timestamp,
      });

      const newRecord: AccessRecord = {
        id: `${Date.now()}-${Math.random()}`,
        driver: recognizedName,
        status: result.authorized ? "authorized" : "unauthorized",
        timestamp,
      };
      setAccessHistory((prev) => [newRecord, ...prev].slice(0, 10));

      if (result.authorized) {
        toast.success(result.message ?? `Motorista autorizado: ${recognizedName}`, {
          description: `Confiança: ${result.confidence?.toFixed?.(1) ?? "N/A"}% · Tempo: ${result.processing_time ?? "N/A"}s`,
          duration: 5000,
        });
      } else {
        toast.error(result.message ?? "Motorista não reconhecido!", {
          description: `Confiança: ${result.confidence?.toFixed?.(1) ?? "N/A"}% · Tempo: ${result.processing_time ?? "N/A"}s`,
          duration: 5000,
        });
      }
    } catch (error) {
      console.error("Erro na verificação:", error);
      toast.error(`Erro ao verificar: ${error instanceof Error ? error.message : "Erro desconhecido"}`);
      setCurrentDriver({ authorized: false, name: "Erro na verificação", timestamp: new Date().toLocaleString("pt-BR") });
    } finally {
      setIsVerifying(false);
    }
  };

  if (!user || isLoadingCar) {
    return (
      <div className="min-h-screen bg-gradient-dark flex items-center justify-center">
        <p className="text-muted-foreground">Carregando...</p>
      </div>
    );
  }

  if (!car) {
    return null;
  }

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
