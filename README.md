# Emotion Detector

Detector de emociones en tiempo real utilizando **MediaPipe Face Mesh**, visión por computador y un sistema heurístico avanzado basado en proporciones geométricas faciales (EAR, MAR y distancia interciliar).

## Características Principales

- **Interfaz Web Moderna (Streamlit):** Panel de control interactivo con diseño en escala de grises y modo dual (Detector en Vivo y Presentación/Diapositivas).
- **Análisis Facial en Tiempo Real:** Detección de 478 puntos tridimensionales mediante *MediaPipe Face Landmarker*.
- **Emociones Soportadas:**
  - Felicidad (basada en elevación de comisuras labiales y apertura de boca)
  - Tristeza (comisuras labiales caídas)
  - Enojo (distancia interciliar reducida y ojos estrechados)
  - Sorpresa (ojos muy abiertos y alta apertura bucal)
  - Neutral
- **Visualización Avanzada:** Indicador gráfico con óvalo facial, flechas guía curvas, superposición de iconos representativos de emociones y visor PiP (Picture-in-Picture) de la malla facial.
- **Script de Consola Alternativo:** Incluye `detector_emociones.py` para ejecución directa con OpenCV.

## Requisitos del Sistema

- **Python:** 3.10 o superior
- **Cámara Web** activa y funcional.

## Instalación

1. Clona el repositorio e ingresa al directorio del proyecto:
   ```bash
   cd Emotion_Detector
   ```

2. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   # En Linux/macOS:
   source venv/bin/activate
   # En Windows:
   venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

### Aplicación Web (Streamlit)
```bash
streamlit run app.py
```

### Script de Consola (OpenCV)
```bash
python detector_emociones.py
```
