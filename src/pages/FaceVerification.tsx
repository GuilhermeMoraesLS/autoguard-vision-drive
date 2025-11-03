import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { CameraCapture } from '@/components/CameraCapture';
import { FaceVerificationVisualizer } from '@/components/FaceVerificationVisualizer';
import { useVerification } from '@/hooks/useVerification';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ArrowLeft, RotateCcw, Shield, AlertTriangle } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';

interface AuthorizedDriver {
  id: string;
  name: string;
  photo_url: string;
}

const FaceVerification = () => {
  const { id } = useParams<{ id: string }>();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [car, setCar] = useState<any>(null);
  const [authorizedDrivers, setAuthorizedDrivers] = useState<AuthorizedDriver[]>([]);
  const [isLoadingCar, setIsLoadingCar] = useState(true);

  const {
    isVerifying,
    verificationResult,
    capturedImage,
    verifyImage,
    clearResults
  } = useVerification({ 
    carId: id || '', 
    authorizedDrivers 
  });

  useEffect(() => {
    const fetchCarAndDrivers = async () => {
      if (!id || !user) return;

      try {
        setIsLoadingCar(true);

        // Buscar dados do carro
        const { data: carData, error: carError } = await supabase
          .from("cars")
          .select("*")
          .eq("id", id)
          .single();

        if (carError) throw carError;

        setCar(carData);

        // Buscar motoristas autorizados
        const { data: driversData, error: driversError } = await supabase
          .from("authorized_drivers")
          .select("id, name, photo_url")
          .eq("car_id", id);

        if (driversError) throw driversError;

        setAuthorizedDrivers(driversData || []);
        
      } catch (error) {
        console.error("Erro ao buscar dados:", error);
        toast.error("Erro ao carregar dados do veículo");
        navigate("/");
      } finally {
        setIsLoadingCar(false);
      }
    };

    fetchCarAndDrivers();
  }, [id, user, navigate]);

  if (!user) {
    navigate("/auth");
    return null;
  }

  if (isLoadingCar) {
    return (
      <div className="min-h-screen bg-gradient-dark flex items-center justify-center">
        <Card className="p-8 text-center">
          <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-muted-foreground">Carregando informações...</p>
        </Card>
      </div>
    );
  }

  if (!car) {
    return (
      <div className="min-h-screen bg-gradient-dark flex items-center justify-center p-4">
        <Card className="p-8 text-center max-w-md w-full">
          <AlertTriangle className="w-12 h-12 text-danger mx-auto mb-4" />
          <h2 className="text-xl font-bold mb-2">Veículo não encontrado</h2>
          <p className="text-muted-foreground mb-4">
            O veículo solicitado não foi encontrado ou você não tem permissão para acessá-lo.
          </p>
          <Button onClick={() => navigate("/")} variant="outline">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Voltar para início
          </Button>
        </Card>
      </div>
    );
  }

  if (authorizedDrivers.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-dark flex items-center justify-center p-4">
        <Card className="p-8 text-center max-w-md w-full">
          <Shield className="w-12 h-12 text-warning mx-auto mb-4" />
          <h2 className="text-xl font-bold mb-2">Nenhum motorista autorizado</h2>
          <p className="text-muted-foreground mb-4">
            Este veículo ainda não possui motoristas autorizados cadastrados.
          </p>
          <div className="space-y-2">
            <Button 
              onClick={() => navigate(`/cars/${id}/drivers/new`)} 
              className="w-full"
            >
              Cadastrar Motorista
            </Button>
            <Button onClick={() => navigate("/")} variant="outline" className="w-full">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Voltar para início
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-dark">
      <div className="container mx-auto p-4 max-w-4xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <Button
            onClick={() => navigate("/")}
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Voltar
          </Button>
          
          <div className="text-center">
            <h1 className="text-xl font-bold text-foreground">
              {car.brand} {car.model}
            </h1>
            <p className="text-sm text-muted-foreground">
              {car.plate} • {authorizedDrivers.length} motorista(s) autorizado(s)
            </p>
          </div>
          
          <div className="w-20" /> {/* Spacer for centering */}
        </div>

        {/* Content */}
        <div className="space-y-6">
          {!verificationResult ? (
            // Tela de Captura
            <div className="space-y-4">
              <Card className="p-6">
                <div className="text-center mb-6">
                  <Shield className="w-12 h-12 text-primary mx-auto mb-4" />
                  <h2 className="text-2xl font-bold mb-2">Verificação Facial</h2>
                  <p className="text-muted-foreground">
                    Posicione-se em frente à câmera para verificar sua identidade
                  </p>
                </div>
                
                <CameraCapture 
                  onCapture={verifyImage} 
                  isVerifying={isVerifying}
                  className="max-w-2xl mx-auto"
                />
              </Card>
            </div>
          ) : (
            // Tela de Resultados
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  Resultado da Verificação
                </h2>
                <Button
                  onClick={clearResults}
                  variant="secondary"
                  size="sm"
                  disabled={isVerifying}
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Nova Verificação
                </Button>
              </div>
              
              <FaceVerificationVisualizer
                imageData={capturedImage}
                verificationResult={verificationResult}
                isLoading={isVerifying}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FaceVerification;