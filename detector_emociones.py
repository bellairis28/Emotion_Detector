import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Verificar y descargar automáticamente el modelo si no existe
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("[INFO] Descargando modelo de MediaPipe Face Landmarker...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("[INFO] ¡Modelo descargado con éxito!")

# Configurar el detector de puntos faciales (FaceLandmarker)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Índices clave de MediaPipe Face Mesh
OJO_IZQ_SUP = 159
OJO_IZQ_INF = 145
OJO_IZQ_EXT = 33
OJO_IZQ_INT = 133

OJO_DER_SUP = 386
OJO_DER_INF = 374
OJO_DER_EXT = 263
OJO_DER_INT = 362

LABIO_SUP = 13
LABIO_INF = 14
COMISURA_IZQ = 61
COMISURA_DER = 291

def calcular_distancia(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calcular_ear(landmarks, sup_idx, inf_idx, ext_idx, int_idx, w, h):
    p_sup = [landmarks[sup_idx].x * w, landmarks[sup_idx].y * h, landmarks[sup_idx].z]
    p_inf = [landmarks[inf_idx].x * w, landmarks[inf_idx].y * h, landmarks[inf_idx].z]
    p_ext = [landmarks[ext_idx].x * w, landmarks[ext_idx].y * h, landmarks[ext_idx].z]
    p_int = [landmarks[int_idx].x * w, landmarks[int_idx].y * h, landmarks[int_idx].z]
    
    vertical = calcular_distancia(p_sup, p_inf)
    horizontal = calcular_distancia(p_ext, p_int)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

def calcular_mar(landmarks, w, h):
    p_sup = [landmarks[LABIO_SUP].x * w, landmarks[LABIO_SUP].y * h, landmarks[LABIO_SUP].z]
    p_inf = [landmarks[LABIO_INF].x * w, landmarks[LABIO_INF].y * h, landmarks[LABIO_INF].z]
    p_izq = [landmarks[COMISURA_IZQ].x * w, landmarks[COMISURA_IZQ].y * h, landmarks[COMISURA_IZQ].z]
    p_der = [landmarks[COMISURA_DER].x * w, landmarks[COMISURA_DER].y * h, landmarks[COMISURA_DER].z]
    
    vertical = calcular_distancia(p_sup, p_inf)
    horizontal = calcular_distancia(p_izq, p_der)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

def clasificar_emocion(ear_promedio, mar, altura_comisuras_relativa):
    """
    Motor de reglas heurísticas con Sorpresa, Felicidad, Tristeza y Enojo.
    """
    if ear_promedio > 0.28 and mar > 0.35:
        return "Sorpresa"
    elif altura_comisuras_relativa > 0.02 and mar > 0.15:
        return "Felicidad"
    elif altura_comisuras_relativa < -0.015 and mar < 0.15:
        return "Tristeza"
    elif ear_promedio < 0.18 and mar < 0.12:
        return "Enojo"
    else:
        return "Neutral"

# Inicializar captura de video
cap = cv2.VideoCapture(0)
print("[INFO] Sistema heurístico de emociones iniciado. Presiona 'q' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            
            ear_izq = calcular_ear(face_landmarks, OJO_IZQ_SUP, OJO_IZQ_INF, OJO_IZQ_EXT, OJO_IZQ_INT, w, h)
            ear_der = calcular_ear(face_landmarks, OJO_DER_SUP, OJO_DER_INF, OJO_DER_EXT, OJO_DER_INT, w, h)
            ear_promedio = (ear_izq + ear_der) / 2.0

            mar = calcular_mar(face_landmarks, w, h)

            # Cálculo de la altura de las comisuras respecto al centro de los labios
            comisura_izq_y = face_landmarks[COMISURA_IZQ].y
            comisura_der_y = face_landmarks[COMISURA_DER].y
            centro_labios_y = (face_landmarks[LABIO_SUP].y + face_landmarks[LABIO_INF].y) / 2.0
            altura_comisuras = centro_labios_y - ((comisura_izq_y + comisura_der_y) / 2.0)

            # Clasificación de la emoción
            emocion = clasificar_emocion(ear_promedio, mar, altura_comisuras)

            # Mostrar métricas y resultado en pantalla
            cv2.putText(frame, f"EAR: {ear_promedio:.2f} | MAR: {mar:.2f}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Comisuras: {altura_comisuras:.3f}", (30, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            cv2.putText(frame, f"Emocion: {emocion}", (30, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Dibujar puntos clave en la malla
            for lm in face_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

    cv2.imshow('Reconocimiento de Emociones - Heuristica', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()