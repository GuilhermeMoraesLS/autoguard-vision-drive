import React, { useState } from 'react';
import { FaceVerificationVisualizer } from './FaceVerificationVisualizer';
import { Button } from './ui/button';

// Dados de teste simulados
const mockImageData = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwDX4AAAD//Z";

const mockResult = {
  detections: [
    {
      authorized: true,
      driver_id: "123",
      driver_name: "João Silva",
      confidence: 85.2,
      x: 50,
      y: 30,
      width: 120,
      height: 150
    },
    {
      authorized: false,
      driver_id: null,
      driver_name: "Desconhecido",
      confidence: 45.7,
      x: 200,
      y: 40,
      width: 110,
      height: 140
    }
  ],
  car_id: "car-123",
  authorized_count: 1,
  unknown_count: 1
};

export const TestVisualizer: React.FC = () => {
  const [showTest, setShowTest] = useState(false);

  return (
    <div className="p-4 space-y-4">
      <Button onClick={() => setShowTest(!showTest)}>
        {showTest ? 'Esconder' : 'Mostrar'} Teste do Visualizador
      </Button>
      
      {showTest && (
        <FaceVerificationVisualizer
          imageData={mockImageData}
          verificationResult={mockResult}
        />
      )}
    </div>
  );
};