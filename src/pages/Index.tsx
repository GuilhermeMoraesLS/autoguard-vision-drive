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

// Interface para a nova estrutura de resposta da API
interface DetectionResult {
  authorized: boolean;
  driver_id: string | null;
  driver_name: string;
  confidence: number;
}

interface ApiResponse {
  status: string;
  message: string;
  detections: DetectionResult[];
  car_id: string;
  authorized_count: number;
  unknown_count: number;
  thresholds: {
    strict: number;
    loose: number;
  };
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
    confidence?: number;
    totalDetections?: number;
    authorizedCount?: number;
    unknownCount?: number;
  }>({
    authorized: null,
    name: "Aguardando...",
    timestamp: "--",
  });
  const [accessHistory, setAccessHistory] = useState<AccessRecord[]>([]);

  useEffect(() => {
    const fetchCarAndDrivers = async () => {
      if (!carId || !user) return;

      try {
        setIsLoadingCar(true);

        // Buscar dados do carro
        const { data: carData, error: carError } = await supabase
          .from("cars")
          .select("*")
          .eq("id", carId)
          .single();

        if (carError) throw carError;

        setCar(carData);

        // Buscar motoristas autorizados
        const { data: driversData, error: driversError } = await supabase
          .from("car_drivers")
          .select(`
            drivers (
              id,
              name,
              photo_url
            )
          `)
          .eq("car_id", carId);

        if (driversError) throw driversError;

        const drivers = driversData
          ?.map((item) => item.drivers)
          .filter(Boolean) as AuthorizedDriver[];

        setAuthorizedDrivers(drivers || []);
      } catch (error) {
        console.error("Erro ao buscar dados:", error);
        toast.error("Erro ao carregar dados do veículo");
        navigate("/");
      } finally {
        setIsLoadingCar(false);
      }
    };

    fetchCarAndDrivers();
  }, [carId, user, navigate]);

  const handleCapture = async (imageData: string) => {
    console.log("🔄 Iniciando verificação...");
    
    console.log("📊 Dados disponíveis:", {
      carId,
      authorizedDriversCount: authorizedDrivers.length,
      backendUrl: import.meta.env.VITE_BACKEND_URL
    });

    if (!carId) {
      console.error("❌ Erro: carId não informado");
      toast.error("Carro não informado");
      return;
    }

    if (!authorizedDrivers.length) {
      console.error("❌ Erro: Nenhum motorista autorizado");
      toast.error("Nenhum motorista autorizado cadastrado para este veículo");
      return;
    }

    setIsVerifying(true);
    toast.info("Processando imagem...");
    setCurrentDriver({ authorized: null, name: "Verificando...", timestamp: "--" });

    try {
      const backendBaseUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
      console.log("🌐 URL do backend:", backendBaseUrl);
      
      const requestData = {
        image: imageData,
        car_id: carId,
        authorized_drivers: authorizedDrivers.map(driver => ({
          id: driver.id,
          name: driver.name,
          photo_url: driver.photo_url
        }))
      };

      console.log("📤 Enviando requisição:", {
        url: `${backendBaseUrl}/verify_driver`,
        driversCount: requestData.authorized_drivers.length,
        carId: requestData.car_id
      });

      const response = await fetch(`${backendBaseUrl}/verify_driver`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      console.log("📥 Resposta recebida:", {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ 
          error: `Erro na API: ${response.status} ${response.statusText}` 
        }));
        throw new Error(errorData.error || `Erro na API: ${response.status} ${response.statusText}`);
      }

      const result: ApiResponse = await response.json();
      console.log("📊 Resultado completo da API:", result);

      // Processar múltiplas detecções
      const timestamp = new Date().toLocaleString("pt-BR");
      
      if (result.detections && result.detections.length > 0) {
        console.log("✅ Detecções encontradas:", result.detections);
        
        // Encontrar a detecção com maior confiança para exibição principal
        const bestDetection = result.detections.reduce((prev, current) => 
          current.confidence > prev.confidence ? current : prev
        );

        // Criar nome descritivo baseado nos resultados
        let statusName = "";
        if (result.authorized_count > 0 && result.unknown_count > 0) {
          statusName = `${result.authorized_count} autorizado(s), ${result.unknown_count} desconhecido(s)`;
        } else if (result.authorized_count > 0) {
          if (result.authorized_count === 1) {
            // Se só há 1 autorizado, mostra o nome específico
            const authorizedDetection = result.detections.find(d => d.authorized);
            statusName = authorizedDetection ? authorizedDetection.driver_name : `${result.authorized_count} autorizado`;
          } else {
            statusName = `${result.authorized_count} autorizados`;
          }
        } else {
          statusName = `${result.unknown_count} desconhecido(s)`;
        }

        // Atualizar status atual
        setCurrentDriver({
          authorized: result.authorized_count > 0,
          name: statusName,
          timestamp,
          confidence: bestDetection.confidence,
          totalDetections: result.detections.length,
          authorizedCount: result.authorized_count,
          unknownCount: result.unknown_count
        });

        // Adicionar TODOS os registros ao histórico (um para cada pessoa detectada)
        const newRecords: AccessRecord[] = result.detections.map((detection, index) => ({
          id: `${Date.now()}-${index}`,
          driver: detection.driver_name,
          status: detection.authorized ? "authorized" : "unauthorized",
          timestamp,
          confidence: detection.confidence
        }));

        // Adicionar no topo do histórico
        setAccessHistory((prev) => [...newRecords, ...prev].slice(0, 50));

        console.log("📋 Novos registros adicionados ao histórico:", newRecords);

        // Mostrar notificação com resumo
        if (result.authorized_count > 0) {
          toast.success(`✅ ${result.authorized_count} motorista(s) autorizado(s) detectado(s)`, {
            description: `${result.unknown_count > 0 ? `⚠️ ${result.unknown_count} desconhecido(s) também detectado(s) • ` : ''}Confiança máxima: ${bestDetection.confidence.toFixed(1)}%`,
            duration: 6000,
          });
        } else {
          toast.error(`❌ ${result.unknown_count} pessoa(s) desconhecida(s) detectada(s)!`, {
            description: `Confiança máxima: ${bestDetection.confidence.toFixed(1)}%`,
            duration: 6000,
          });
        }
      } else {
        // Nenhuma face detectada
        console.log("❌ Nenhuma face detectada");
        setCurrentDriver({
          authorized: false,
          name: "Nenhuma face detectada",
          timestamp,
          totalDetections: 0,
          authorizedCount: 0,
          unknownCount: 0
        });

        toast.error("❌ Nenhum rosto detectado na imagem", {
          description: "Certifique-se de que há uma pessoa visível na captura",
          duration: 5000,
        });
      }

    } catch (error) {
      console.error("❌ Erro na verificação:", error);
      console.error("📋 Detalhes do erro:", {
        message: error instanceof Error ? error.message : "Erro desconhecido",
        stack: error instanceof Error ? error.stack : undefined
      });
      
      toast.error(`❌ Erro ao verificar: ${error instanceof Error ? error.message : "Erro desconhecido"}`);
      setCurrentDriver({ 
        authorized: false, 
        name: "Erro na verificação", 
        timestamp: new Date().toLocaleString("pt-BR") 
      });
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
                <p className="text-sm text-muted-foreground">
                  Verificação de Motorista - {car?.brand} {car?.model} ({car?.plate})
                </p>
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
                Posicione o(s) motorista(s) em frente à câmera e clique em "Verificar Motorista"
              </p>
            </div>
            <CameraCapture onCapture={handleCapture} isVerifying={isVerifying} />
          </div>

          {/* Right Column - Status & Info */}
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
                confidence={currentDriver.confidence}
                totalDetections={currentDriver.totalDetections}
                authorizedCount={currentDriver.authorizedCount}
                unknownCount={currentDriver.unknownCount}
              />
            </div>

            {/* Motoristas Autorizados */}
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-2">
                Motoristas Autorizados ({authorizedDrivers.length})
              </h3>
              {authorizedDrivers.length > 0 ? (
                <div className="space-y-2">
                  {authorizedDrivers.slice(0, 3).map((driver) => (
                    <div key={driver.id} className="flex items-center gap-2 p-2 bg-secondary/30 rounded-lg">
                      <img 
                        src={driver.photo_url} 
                        alt={driver.name}
                        className="w-8 h-8 rounded-full object-cover"
                      />
                      <span className="text-sm text-foreground">{driver.name}</span>
                    </div>
                  ))}
                  {authorizedDrivers.length > 3 && (
                    <p className="text-xs text-muted-foreground">
                      +{authorizedDrivers.length - 3} outros...
                    </p>
                  )}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Nenhum motorista cadastrado</p>
              )}
            </div>
          </div>
        </div>

        {/* History Table - Full Width */}
        <div className="mt-8">
          <AccessHistory records={accessHistory} />
        </div>
      </main>
    </div>
  );
};

export default Index;
