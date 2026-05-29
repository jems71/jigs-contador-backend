import cv2
import numpy as np
import uvicorn
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="Motor de Conteo JIGS")

# Permite que tu página web de Vercel se comunique con Railway sin bloqueos de seguridad
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite conexiones desde cualquier origen (ideal para el celular)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargamos tu modelo entrenado de 150 épocas directamente en la memoria del servidor
try:
    model = YOLO("modelo_jigs_marcas_x.pt")
    print("Cerebro YOLOv8 cargado con éxito en terreno.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

@app.get("/")
def estado_servidor():
    return {"status": "Servidor JIGS Operativo en Railway", "modelo": "YOLOv8_150_epochs"}

@app.post("/api/v1/count")
async def procesar_foto_terreno(file: UploadFile = File(...)):
    try:
        # 1. Leer los bytes de la foto que mandó el celular
        file_bytes = await file.read()
        
        # 2. Convertir los bytes en una matriz de imagen real (OpenCV)
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "No se pudo decodificar la imagen", "total": 0}
            
        # 3. Ejecutar la Inteligencia Artificial con tu umbral de confianza calibrado
        results = model.predict(source=img, conf=0.25, verbose=False)
        
        # 4. Cubicar las cajas (Contar cuántas marcas X detectó)
        coordenadas_cajas = results[0].boxes.xyxy.tolist()
        total_marcas = len(coordenadas_cajas)
        
        # Devolvemos el resultado limpio al celular
        return {
            "success": True,
            "total": total_marcas,
            "engine": "JIGS-Proprietary-AI"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "total": 0}

# ESTO DEBE IR FUERA DE LA FUNCIÓN (SIN SANGRÍA)
if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=puerto)
