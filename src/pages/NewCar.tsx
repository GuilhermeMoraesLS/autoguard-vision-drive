import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Shield, ArrowLeft, Save } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { z } from "zod";

const carSchema = z.object({
  brand: z.string().trim().min(1, "Marca é obrigatória").max(50, "Marca muito longa"),
  model: z.string().trim().min(1, "Modelo é obrigatório").max(50, "Modelo muito longo"),
  plate: z.string().trim().min(7, "Placa inválida").max(8, "Placa inválida"),
  year: z.number().min(1900, "Ano inválido").max(new Date().getFullYear() + 1, "Ano inválido").optional(),
});

const NewCar = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  
  const [formData, setFormData] = useState({
    brand: "",
    model: "",
    plate: "",
    year: "",
  });

  if (!user) {
    navigate("/auth");
    return null;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrors({});
    setIsLoading(true);

    try {
      const yearValue = formData.year ? parseInt(formData.year) : undefined;
      
      const result = carSchema.safeParse({
        ...formData,
        year: yearValue,
      });

      if (!result.success) {
        const fieldErrors: Record<string, string> = {};
        result.error.issues.forEach((issue) => {
          if (issue.path[0]) {
            fieldErrors[issue.path[0].toString()] = issue.message;
          }
        });
        setErrors(fieldErrors);
        setIsLoading(false);
        return;
      }

      const { error } = await supabase.from("cars").insert({
        user_id: user.id,
        brand: result.data.brand,
        model: result.data.model,
        plate: result.data.plate.toUpperCase(),
        year: result.data.year,
      });

      if (error) {
        if (error.message.includes("duplicate key")) {
          toast.error("Você já cadastrou um carro com esta placa");
        } else {
          toast.error("Erro ao cadastrar carro");
        }
        console.error("Insert error:", error);
      } else {
        toast.success("Carro cadastrado com sucesso!");
        navigate("/");
      }
    } catch (error) {
      console.error("Error:", error);
      toast.error("Erro ao cadastrar carro");
    } finally {
      setIsLoading(false);
    }
  };

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
              <p className="text-sm text-muted-foreground">Adicionar Carro</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-2xl">
        <Button
          onClick={() => navigate("/")}
          variant="secondary"
          className="mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Voltar
        </Button>

        <Card className="p-8 bg-card border-border">
          <h2 className="text-2xl font-bold text-foreground mb-6">Cadastrar Novo Carro</h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="brand">Marca *</Label>
              <Input
                id="brand"
                value={formData.brand}
                onChange={(e) => setFormData({ ...formData, brand: e.target.value })}
                className="bg-secondary border-border"
                placeholder="Ex: Toyota"
                disabled={isLoading}
              />
              {errors.brand && (
                <p className="text-sm text-danger">{errors.brand}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="model">Modelo *</Label>
              <Input
                id="model"
                value={formData.model}
                onChange={(e) => setFormData({ ...formData, model: e.target.value })}
                className="bg-secondary border-border"
                placeholder="Ex: Corolla"
                disabled={isLoading}
              />
              {errors.model && (
                <p className="text-sm text-danger">{errors.model}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="plate">Placa *</Label>
              <Input
                id="plate"
                value={formData.plate}
                onChange={(e) => setFormData({ ...formData, plate: e.target.value })}
                className="bg-secondary border-border"
                placeholder="Ex: ABC1234"
                maxLength={8}
                disabled={isLoading}
              />
              {errors.plate && (
                <p className="text-sm text-danger">{errors.plate}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="year">Ano</Label>
              <Input
                id="year"
                type="number"
                value={formData.year}
                onChange={(e) => setFormData({ ...formData, year: e.target.value })}
                className="bg-secondary border-border"
                placeholder="Ex: 2020"
                disabled={isLoading}
              />
              {errors.year && (
                <p className="text-sm text-danger">{errors.year}</p>
              )}
            </div>

            <Button
              type="submit"
              className="w-full bg-gradient-primary hover:opacity-90 transition-opacity"
              size="lg"
              disabled={isLoading}
            >
              <Save className="w-5 h-5 mr-2" />
              {isLoading ? "Salvando..." : "Salvar Carro"}
            </Button>
          </form>
        </Card>
      </main>
    </div>
  );
};

export default NewCar;
