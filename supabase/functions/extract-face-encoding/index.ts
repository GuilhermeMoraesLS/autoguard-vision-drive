import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { photo_url } = await req.json();
    
    if (!photo_url) {
      return new Response(
        JSON.stringify({ error: "URL da foto é obrigatória" }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log(`Processando foto: ${photo_url}`);

    // Baixar a imagem do Supabase Storage
    const imageResponse = await fetch(photo_url);
    if (!imageResponse.ok) {
      console.error(`Erro ao baixar imagem: ${imageResponse.status}`);
      return new Response(
        JSON.stringify({ error: "Erro ao baixar a foto" }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const imageBuffer = await imageResponse.arrayBuffer();
    const base64Image = btoa(
      new Uint8Array(imageBuffer).reduce((data, byte) => data + String.fromCharCode(byte), '')
    );

    // Chamar backend Python para extrair encoding
    const backendUrl = Deno.env.get('VITE_BACKEND_URL') || 'http://localhost:8000';
    
    console.log(`Chamando backend Python: ${backendUrl}/extract_encoding`);
    
    const pythonResponse = await fetch(`${backendUrl}/extract_encoding`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: `data:image/jpeg;base64,${base64Image}`
      }),
    });

    if (!pythonResponse.ok) {
      const errorText = await pythonResponse.text();
      console.error(`Erro do backend Python: ${pythonResponse.status} - ${errorText}`);
      return new Response(
        JSON.stringify({ error: "Erro ao processar a foto no backend" }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const result = await pythonResponse.json();
    
    console.log(`Resultado do encoding: ${JSON.stringify(result)}`);

    // Verificar se encontrou um rosto
    if (!result.face_encoding) {
      return new Response(
        JSON.stringify({ 
          error: "Nenhum rosto detectado na foto. Por favor, tire outra foto com o rosto bem visível.",
          no_face: true 
        }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Retornar o encoding como string JSON
    return new Response(
      JSON.stringify({ 
        face_encoding: JSON.stringify(result.face_encoding),
        success: true 
      }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Erro ao extrair encoding:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Erro desconhecido" }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
