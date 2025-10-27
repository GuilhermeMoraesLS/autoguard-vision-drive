import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Shield, ArrowLeft, Save, Upload, Camera, Image } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { z } from "zod";
import { PhotoCapture } from "@/components/PhotoCapture";

const driverSchema = z.object({
  name: z.string().trim().min(2, "Nome deve ter no mínimo 2 caracteres").max(100, "Nome muito longo"),
});

type PhotoMode = 'none' | 'upload' | 'camera';

const NewDriver = () => {
  const { id } = useParams<{ id: string }>();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  
  const [name, setName] = useState("");
  const [photoFile, setPhotoFile] = useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string>("");
  const [photoMode, setPhotoMode] = useState<PhotoMode>('none');

  if (!user || !id) {
    navigate("/auth");
    return null;
  }

  const handlePhotoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith("image/")) {
      toast.error("Por favor, selecione uma imagem");
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      toast.error("A imagem deve ter no máximo 5MB");
      return;
    }

    setPhotoFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPhotoPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
    setPhotoMode('upload');
  };

  const handlePhotoCapture = (photoBlob: Blob) => {
    // Converte blob para file
    const file = new File([photoBlob], `foto-${Date.now()}.jpg`, { type: 'image/jpeg' });
    setPhotoFile(file);
    
    // Cria preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPhotoPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
    
    setPhotoMode('upload');
    toast.success("Foto capturada com sucesso!");
  };

  const resetPhoto = () => {
    setPhotoFile(null);
    setPhotoPreview("");
    setPhotoMode('none');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrors({});

    if (!photoFile) {
      toast.error("Por favor, selecione ou tire uma foto do motorista");
      return;
    }

    setIsLoading(true);

    try {
      const result = driverSchema.safeParse({ name });

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

      // Upload photo to storage
      const fileExt = photoFile.name.split(".").pop();
      const fileName = `${user.id}/${Date.now()}.${fileExt}`;
      
      const { error: uploadError } = await supabase.storage
        .from("driver-photos")
        .upload(fileName, photoFile);

      if (uploadError) {
        console.error("Upload error:", uploadError);
        toast.error("Erro ao fazer upload da foto");
        setIsLoading(false);
        return;
      }

      // Get public URL
      const { data: { publicUrl } } = supabase.storage
        .from("driver-photos")
        .getPublicUrl(fileName);

      // Insert driver record
      const { error: insertError } = await supabase
        .from("authorized_drivers")
        .insert({
          car_id: id,
          name: result.data.name,
          photo_url: publicUrl,
        });

      if (insertError) {
        console.error("Insert error:", insertError);
        toast.error("Erro ao cadastrar motorista");
      } else {
        toast.success("Motorista cadastrado com sucesso!");
        navigate(`/cars/${id}`);
      }
    } catch (error) {
      console.error("Error:", error);
      toast.error("Erro ao cadastrar motorista");
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
              <p className="text-sm text-muted-foreground">Adicionar Motorista</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-2xl">
        <Button
          onClick={() => navigate(`/cars/${id}`)}
          variant="secondary"
          className="mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Voltar
        </Button>

        <Card className="p-8 bg-card border-border">
          <h2 className="text-2xl font-bold text-foreground mb-6">Cadastrar Motorista Autorizado</h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="name">Nome Completo *</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="bg-secondary border-border"
                placeholder="Ex: João Silva"
                disabled={isLoading}
              />
              {errors.name && (
                <p className="text-sm text-danger">{errors.name}</p>
              )}
            </div>

            <div className="space-y-4">
              <Label>Foto do Rosto *</Label>
              
              {photoMode === 'none' && (
                <div className="grid grid-cols-2 gap-4">
                  <Button
                    type="button"
                    variant="secondary"
                    className="h-32 flex-col gap-3"
                    onClick={() => setPhotoMode('camera')}
                    disabled={isLoading}
                  >
                    <Camera className="w-8 h-8" />
                    <div className="text-center">
                      <p className="font-medium">Tirar Foto</p>
                      <p className="text-xs text-muted-foreground">Usar câmera do dispositivo</p>
                    </div>
                  </Button>
                  
                  <label className="h-32 flex flex-col items-center justify-center gap-3 bg-secondary rounded-lg border-2 border-dashed border-border cursor-pointer hover:bg-secondary/80 transition-colors">
                    <Upload className="w-8 h-8" />
                    <div className="text-center">
                      <p className="font-medium text-sm">Carregar Foto</p>
                      <p className="text-xs text-muted-foreground">PNG, JPG até 5MB</p>
                    </div>
                    <input
                      type="file"
                      className="hidden"
                      accept="image/*"
                      onChange={handlePhotoUpload}
                      disabled={isLoading}
                    />
                  </label>
                </div>
              )}

              {photoMode === 'camera' && (
                <PhotoCapture
                  onCapture={handlePhotoCapture}
                  onCancel={() => setPhotoMode('none')}
                />
              )}

              {photoMode === 'upload' && photoPreview && (
                <div className="space-y-4">
                  <div className="relative w-full aspect-video bg-secondary rounded-lg overflow-hidden border-2 border-border">
                    <img
                      src={photoPreview}
                      alt="Preview da foto"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      variant="secondary"
                      onClick={resetPhoto}
                      disabled={isLoading}
                      className="flex-1"
                    >
                      <ArrowLeft className="w-4 h-4 mr-2" />
                      Escolher Outra
                    </Button>
                  </div>
                </div>
              )}

              <p className="text-xs text-muted-foreground">
                Tire ou carregue uma foto clara do rosto da pessoa para reconhecimento facial
              </p>
            </div>

            <Button
              type="submit"
              className="w-full bg-gradient-primary hover:opacity-90 transition-opacity"
              size="lg"
              disabled={isLoading || !photoFile}
            >
              <Save className="w-5 h-5 mr-2" />
              {isLoading ? "Salvando..." : "Salvar Motorista"}
            </Button>
          </form>
        </Card>

        <div className="mt-6 p-6 bg-accent/10 border border-accent/30 rounded-lg">
          <h3 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
            <Camera className="w-4 h-4" />
            📸 Dicas para melhor reconhecimento
          </h3>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Use boa iluminação natural ou artificial</li>
            <li>• Tire a foto de frente para a câmera</li>
            <li>• Evite óculos escuros ou chapéus</li>
            <li>• Mantenha expressão neutra</li>
            <li>• Certifique-se que o rosto esteja bem enquadrado</li>
          </ul>
        </div>
      </main>
    </div>
  );
};

export default NewDriver;
