import cv2
import time
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
# Ojos
OJO_IZQ_SUP = 159
OJO_IZQ_INF = 145
OJO_IZQ_EXT = 33
OJO_IZQ_INT = 133

OJO_DER_SUP = 386
OJO_DER_INF = 374
OJO_DER_EXT = 263
OJO_DER_INT = 362

# Boca
LABIO_SUP = 13
LABIO_INF = 14
COMISURA_IZQ = 61
COMISURA_DER = 291

# Cejas (puntos internos más cercanos a la glabela)
CEJA_IZQ_INT = 55
CEJA_DER_INT = 285

# Umbrales para detectar "Dormido"
EAR_OJOS_CERRADOS = 0.15      # EAR por debajo de esto = ojos cerrados
SEGUNDOS_PARA_DORMIDO = 1.5   # tiempo continuo con ojos cerrados

# Calibración y umbrales para "Tristeza"
FRAMES_CALIBRACION = 40
UMBRAL_CAIDA_COMISURAS = 0.035   # cuánto deben bajar las comisuras vs. reposo (fracción de dist. interocular)
UMBRAL_CENO = 0.025              # cuánto deben acercarse las cejas vs. reposo

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

def clasificar_emocion(ear_promedio, mar, altura_comisuras_relativa, distancia_cejas_norm,
                       caida_comisuras, cierre_cejas, segundos_ojos_cerrados=0.0):
    """
    Motor de reglas heurísticas. Tristeza usa cambios respecto a la cara neutra calibrada.
    """
    # 0. Dormido: ojos cerrados de forma sostenida (prioridad máxima)
    if segundos_ojos_cerrados >= SEGUNDOS_PARA_DORMIDO:
        return "Dormido"

    # 1. Sorpresa: Ojos muy abiertos y boca abierta
    elif ear_promedio > 0.28 and mar > 0.35:
        return "Sorpresa"
    
    # 2. Felicidad: Comisuras de los labios elevadas
    elif altura_comisuras_relativa > 0.02 and mar > 0.15:
        return "Felicidad"
    
    # 3. Tristeza: comisuras caídas + ceño fruncido (respecto a tu cara en reposo)
    elif (caida_comisuras > UMBRAL_CAIDA_COMISURAS
          and cierre_cejas > UMBRAL_CENO
          and mar < 0.20):
        return "Tristeza"
    
    # 4. Enojo: Cejas notablemente juntas (distancia baja) + ojos ligeramente estrechados
    #    (se añade el límite inferior para que un parpadeo no cuente como Enojo)
    elif distancia_cejas_norm < 0.35 and EAR_OJOS_CERRADOS <= ear_promedio < 0.22:
        return "Enojo"
    
    # 5. Estado base
    else:
        return "Neutral"

# Inicializar captura de video
cap = cv2.VideoCapture(0)
ojos_cerrados_desde = None

# Estado de calibración
calib_alt = []
calib_cejas = []
base_alt = None
base_cejas = None

print("[INFO] Sistema iniciado. Mantén cara neutra ~1-2 s para calibrar. Presiona 'q' para salir.")

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
            
            # 1. Calcular métricas oculares y bucales
            ear_izq = calcular_ear(face_landmarks, OJO_IZQ_SUP, OJO_IZQ_INF, OJO_IZQ_EXT, OJO_IZQ_INT, w, h)
            ear_der = calcular_ear(face_landmarks, OJO_DER_SUP, OJO_DER_INF, OJO_DER_EXT, OJO_DER_INT, w, h)
            ear_promedio = (ear_izq + ear_der) / 2.0

            # Cronómetro de ojos cerrados
            if ear_promedio < EAR_OJOS_CERRADOS:
                if ojos_cerrados_desde is None:
                    ojos_cerrados_desde = time.time()
                segundos_cerrados = time.time() - ojos_cerrados_desde
            else:
                ojos_cerrados_desde = None
                segundos_cerrados = 0.0

            mar = calcular_mar(face_landmarks, w, h)

            # 2. Altura de comisuras labiales
            comisura_izq_y = face_landmarks[COMISURA_IZQ].y
            comisura_der_y = face_landmarks[COMISURA_DER].y
            centro_labios_y = (face_landmarks[LABIO_SUP].y + face_landmarks[LABIO_INF].y) / 2.0
            altura_comisuras = centro_labios_y - ((comisura_izq_y + comisura_der_y) / 2.0)

            # 3. Distancia entre las cejas normalizada por la distancia interocular
            ceja_izq = [face_landmarks[CEJA_IZQ_INT].x * w, face_landmarks[CEJA_IZQ_INT].y * h]
            ceja_der = [face_landmarks[CEJA_DER_INT].x * w, face_landmarks[CEJA_DER_INT].y * h]
            ojo_ext_izq = [face_landmarks[OJO_IZQ_EXT].x * w, face_landmarks[OJO_IZQ_EXT].y * h]
            ojo_ext_der = [face_landmarks[OJO_DER_EXT].x * w, face_landmarks[OJO_DER_EXT].y * h]
            
            dist_cejas = calcular_distancia(ceja_izq, ceja_der)
            dist_interocular = calcular_distancia(ojo_ext_izq, ojo_ext_der)
            
            distancia_cejas_norm = dist_cejas / dist_interocular if dist_interocular > 0 else 0.0

            # Altura de comisuras normalizada (positivo = comisuras arriba del centro de los labios)
            altura_norm = (altura_comisuras * h) / dist_interocular if dist_interocular > 0 else 0.0

            # 4. Calibración con cara neutra y clasificación
            if base_alt is None:
                calib_alt.append(altura_norm)
                calib_cejas.append(distancia_cejas_norm)
                if len(calib_alt) >= FRAMES_CALIBRACION:
                    base_alt = float(np.median(calib_alt))
                    base_cejas = float(np.median(calib_cejas))
                emocion = "Neutral"
                cv2.putText(frame, "Calibrando: manten cara neutra...", (30, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            else:
                caida = base_alt - altura_norm                 # >0 = comisuras más bajas que en reposo
                cierre = base_cejas - distancia_cejas_norm     # >0 = cejas más juntas que en reposo
                emocion = clasificar_emocion(ear_promedio, mar, altura_comisuras, distancia_cejas_norm,
                                             caida, cierre, segundos_cerrados)
                # Valores de depuración para ajustar umbrales
                cv2.putText(frame, f"caida: {caida:.3f}  ceno: {cierre:.3f}", (30, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

            # 5. Mostrar métricas en pantalla
            cv2.putText(frame, f"EAR: {ear_promedio:.2f} | MAR: {mar:.2f}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Cejas Norm: {distancia_cejas_norm:.2f}", (30, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            cv2.putText(frame, f"Emocion: {emocion}", (30, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Dibujar puntos clave
            for lm in face_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
    else:
        # Sin rostro: reinicia el cronómetro de ojos cerrados
        ojos_cerrados_desde = None

    cv2.imshow('Reconocimiento de Emociones - Heuristica', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Recalibrar con la tecla 'r'
        calib_alt.clear()
        calib_cejas.clear()
        base_alt = None
        base_cejas = None

cap.release()
cv2.destroyAllWindows()