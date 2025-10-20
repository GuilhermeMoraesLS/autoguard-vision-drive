import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Shield, ArrowLeft, Plus, Trash2, User } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

interface Car {
  id: string;
  brand: string;
  model: string;
  plate: string;
  year: number | null;
}

interface AuthorizedDriver {
  id: string;
  name: string;
  photo_url: string;
  created_at: string;
}

const CarDetails = () => {
  const { id } = useParams<{ id: string }>();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [car, setCar] = useState<Car | null>(null);
  const [drivers, setDrivers] = useState<AuthorizedDriver[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) {
      navigate("/auth");
      return;
    }
    if (id) {
      fetchCarDetails();
    }
  }, [user, id, navigate]);

  const fetchCarDetails = async () => {
    try {
      const { data: carData, error: carError } = await supabase
        .from("cars")
        .select("*")
        .eq("id", id)
        .single();

      if (carError) throw carError;
      setCar(carData);

      const { data: driversData, error: driversError } = await supabase
        .from("authorized_drivers")
        .select("*")
        .eq("car_id", id)
        .order("created_at", { ascending: false });

      if (driversError) throw driversError;
      setDrivers(driversData || []);
    } catch (error) {
      console.error("Error fetching car details:", error);
      toast.error("Erro ao carregar dados do carro");
      navigate("/");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteDriver = async (driverId: string) => {
    if (!confirm("Tem certeza que deseja remover este motorista?")) return;

    try {
      const { error } = await supabase
        .from("authorized_drivers")
        .delete()
        .eq("id", driverId);

      if (error) throw error;
      
      toast.success("Motorista removido com sucesso");
      fetchCarDetails();
    } catch (error) {
      console.error("Error deleting driver:", error);
      toast.error("Erro ao remover motorista");
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-dark flex items-center justify-center">
        <p className="text-muted-foreground">Carregando...</p>
      </div>
    );
  }

  if (!car) return null;

  return (
    <div className="min-h-screen bg-gradient-dark">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center shadow-glow-primary">
              <Shield className="w-7 h-7 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">AutoGuard Vision Web</h1>
              <p className="text-sm text-muted-foreground">Detalhes do Carro</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <Button
          onClick={() => navigate("/")}
          variant="secondary"
          className="mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Voltar
        </Button>

        <Card className="p-6 bg-card border-border mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-4">
            {car.brand} {car.model}
          </h2>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Placa</p>
              <p className="text-foreground font-semibold">{car.plate}</p>
            </div>
            {car.year && (
              <div>
                <p className="text-muted-foreground">Ano</p>
                <p className="text-foreground font-semibold">{car.year}</p>
              </div>
            )}
          </div>
        </Card>

        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-xl font-bold text-foreground">Motoristas Autorizados</h3>
            <p className="text-sm text-muted-foreground">
              Gerencie quem pode dirigir este veículo
            </p>
          </div>
          <Button
            onClick={() => navigate(`/cars/${id}/drivers/new`)}
            className="bg-gradient-primary hover:opacity-90 transition-opacity"
          >
            <Plus className="w-5 h-5 mr-2" />
            Adicionar Motorista
          </Button>
        </div>

        {drivers.length === 0 ? (
          <Card className="p-12 text-center bg-card border-border">
            <User className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-xl font-semibold text-foreground mb-2">
              Nenhum motorista cadastrado
            </h3>
            <p className="text-muted-foreground mb-6">
              Adicione motoristas autorizados para este veículo
            </p>
            <Button
              onClick={() => navigate(`/cars/${id}/drivers/new`)}
              className="bg-gradient-primary hover:opacity-90 transition-opacity"
            >
              <Plus className="w-5 h-5 mr-2" />
              Adicionar Motorista
            </Button>
          </Card>
        ) : (
          <div className="grid md:grid-cols-2 gap-6">
            {drivers.map((driver) => (
              <Card key={driver.id} className="p-6 bg-card border-border">
                <div className="flex items-start gap-4">
                  <img
                    src={driver.photo_url}
                    alt={driver.name}
                    className="w-16 h-16 rounded-lg object-cover"
                  />
                  <div className="flex-1">
                    <h4 className="text-lg font-semibold text-foreground">{driver.name}</h4>
                    <p className="text-xs text-muted-foreground">
                      Cadastrado em {new Date(driver.created_at).toLocaleDateString("pt-BR")}
                    </p>
                    <Button
                      onClick={() => handleDeleteDriver(driver.id)}
                      variant="secondary"
                      size="sm"
                      className="mt-3"
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Remover
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default CarDetails;
