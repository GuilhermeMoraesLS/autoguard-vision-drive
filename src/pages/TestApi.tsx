import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle, XCircle, Loader2 } from "lucide-react";

const TestApi = () => {
  const [apiStatus, setApiStatus] = useState<{
    status: string;
    message: string;
    loading: boolean;
    error: string | null;
  }>({
    status: "unknown",
    message: "Verificando...",
    loading: true,
    error: null
  });

  const checkApiHealth = async () => {
    setApiStatus({ ...apiStatus, loading: true, error: null });
    
    try {
      const response = await fetch('http://localhost:8000/health');
      
      if (response.ok) {
        const data = await response.json();
        setApiStatus({
          status: "online",
          message: data.message || "API funcionando",
          loading: false,
          error: null
        });
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      setApiStatus({
        status: "offline",
        message: "API não está respondendo",
        loading: false,
        error: error instanceof Error ? error.message : "Erro desconhecido"
      });
    }
  };

  useEffect(() => {
    checkApiHealth();
  }, []);

  const testRecognition = async () => {
    try {
      // Simula uma imagem base64 de teste
      const testImage = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=";
      
      const response = await fetch('http://localhost:8000/verify_driver', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: testImage,
          authorized_drivers: [
            {
              id: "test-1",
              name: "Motorista Teste",
              photo_url: "https://via.placeholder.com/150"
            }
          ]
        })
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Teste bem-sucedido!\nResultado: ${result.message}\nConfiança: ${result.confidence}%`);
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      alert(`Erro no teste: ${error instanceof Error ? error.message : "Erro desconhecido"}`);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-2xl">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            🧪 Teste da API de Reconhecimento Facial
          </CardTitle>
          <CardDescription>
            Verificar se o backend Python está funcionando corretamente
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Status da API */}
          <div className="flex items-center justify-between p-4 border rounded-lg">
            <div>
              <h3 className="font-medium">Status da API</h3>
              <p className="text-sm text-muted-foreground">{apiStatus.message}</p>
              {apiStatus.error && (
                <p className="text-sm text-red-500">Erro: {apiStatus.error}</p>
              )}
            </div>
            <div className="flex items-center gap-2">
              {apiStatus.loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : apiStatus.status === "online" ? (
                <CheckCircle className="w-5 h-5 text-green-500" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
              <Badge variant={apiStatus.status === "online" ? "default" : "destructive"}>
                {apiStatus.status === "online" ? "Online" : "Offline"}
              </Badge>
            </div>
          </div>

          {/* Botões de teste */}
          <div className="space-y-2">
            <Button 
              onClick={checkApiHealth} 
              variant="secondary" 
              className="w-full"
              disabled={apiStatus.loading}
            >
              {apiStatus.loading ? "Verificando..." : "🔄 Verificar API Novamente"}
            </Button>

            <Button 
              onClick={testRecognition} 
              className="w-full"
              disabled={apiStatus.status !== "online"}
            >
              🎯 Testar Reconhecimento Facial
            </Button>
          </div>

          {/* Instruções */}
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">📋 Instruções:</h4>
            <ol className="text-sm space-y-1 list-decimal list-inside">
              <li>Certifique-se que o backend Python está rodando na porta 8000</li>
              <li>Clique em "Verificar API" para testar a conexão</li>
              <li>Se estiver online, clique em "Testar Reconhecimento" para simular uma verificação</li>
              <li>Se tudo estiver funcionando, vá para <a href="/verify" className="text-blue-500 underline">/verify</a> para testar com a câmera</li>
            </ol>
          </div>

          {/* Links úteis */}
          <div className="flex gap-2">
            <Button variant="outline" asChild className="flex-1">
              <a href="http://localhost:8000" target="_blank" rel="noopener noreferrer">
                🌐 Abrir API Backend
              </a>
            </Button>
            <Button variant="outline" asChild className="flex-1">
              <a href="/verify">
                📸 Ir para Verificação
              </a>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TestApi;