import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Shield, LogOut, Plus, Car as CarIcon } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

interface Car {
  id: string;
  brand: string;
  model: string;
  plate: string;
  year: number | null;
}

const Dashboard = () => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [cars, setCars] = useState<Car[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) {
      navigate("/auth");
      return;
    }
    fetchCars();
  }, [user, navigate]);

  const fetchCars = async () => {
    try {
      const { data, error } = await supabase
        .from("cars")
        .select("*")
        .order("created_at", { ascending: false });

      if (error) throw error;
      setCars(data || []);
    } catch (error) {
      console.error("Error fetching cars:", error);
      toast.error("Erro ao carregar carros");
    } finally {
      setLoading(false);
    }
  };

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
                <p className="text-sm text-muted-foreground">Painel de Controle</p>
              </div>
            </div>
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
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-foreground">Meus Carros</h2>
            <p className="text-sm text-muted-foreground">
              Gerencie seus veículos e motoristas autorizados
            </p>
          </div>
          <Button
            onClick={() => navigate("/cars/new")}
            className="bg-gradient-primary hover:opacity-90 transition-opacity"
          >
            <Plus className="w-5 h-5 mr-2" />
            Adicionar Carro
          </Button>
        </div>

        {loading ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground">Carregando...</p>
          </div>
        ) : cars.length === 0 ? (
          <Card className="p-12 text-center bg-card border-border">
            <CarIcon className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-xl font-semibold text-foreground mb-2">
              Nenhum carro cadastrado
            </h3>
            <p className="text-muted-foreground mb-6">
              Adicione seu primeiro veículo para começar
            </p>
            <Button
              onClick={() => navigate("/cars/new")}
              className="bg-gradient-primary hover:opacity-90 transition-opacity"
            >
              <Plus className="w-5 h-5 mr-2" />
              Adicionar Carro
            </Button>
          </Card>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {cars.map((car) => (
              <Card
                key={car.id}
                className="p-6 bg-card border-border hover:border-primary transition-colors cursor-pointer"
                onClick={() => navigate(`/cars/${car.id}`)}
              >
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-lg bg-secondary flex items-center justify-center">
                    <CarIcon className="w-6 h-6 text-primary" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-foreground">
                      {car.brand} {car.model}
                    </h3>
                    <p className="text-sm text-muted-foreground">{car.plate}</p>
                    {car.year && (
                      <p className="text-xs text-muted-foreground mt-1">{car.year}</p>
                    )}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* Quick access to verification */}
        {cars.length > 0 && (
          <div className="mt-8">
            <Button
              onClick={() => navigate("/verify")}
              className="w-full md:w-auto bg-gradient-primary hover:opacity-90 transition-opacity shadow-glow-primary"
              size="lg"
            >
              Iniciar Verificação de Motorista
            </Button>
          </div>
        )}
      </main>
    </div>
  );
};

export default Dashboard;
