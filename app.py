import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import time

# Configuración de la página
st.set_page_config(
    page_title="Detector de Emociones en Tiempo Real",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS avanzados para un diseño ultra moderno en escala de grises
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #171717 100%);
        color: #f5f5f7;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 2rem;
        background: rgba(26, 26, 26, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #8e8e93;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }
    
    .badge {
        background: #2c2c2e;
        color: #e5e5e7;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .glass-card {
        background: rgba(22, 22, 24, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.25rem;
    }
    
    .emotion-pill {
        padding: 0.85rem 1.25rem;
        border-radius: 0.75rem;
        font-weight: 700;
        font-size: 1.35rem;
        text-align: center;
        background: #242426;
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }
    </style>
    
    <div class="header-container">
        <div>
            <h1 class="main-title">Detector de Emociones en Tiempo Real</h1>
            <p class="subtitle">Sistema de análisis facial con MediaPipe Mesh y Visión por Computador</p>
        </div>
        <div>
            <span class="badge">
                <span style="width: 8px; height: 8px; background: #86868b; border-radius: 50%; display: inline-block;"></span>
                Sistema Activo
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Descargar modelo si no existe
MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# Cargar imágenes de emociones para la esquina superior izquierda (140x140)
EMOTION_IMAGES = {}
emotion_files = {
    "Felicidad": "imagenes/happy.jpeg",
    "Tristeza": "imagenes/sad.jpeg",
    "Enojo": "imagenes/angry.jpeg",
    "Sorpresa": "imagenes/surprise.jpeg"
}
for emo, path in emotion_files.items():
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            EMOTION_IMAGES[emo] = cv2.resize(img, (140, 140))

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
    # Parámetros originales restaurados exactamente como estaban
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

def draw_curved_arrow(img, start, end, color=(0, 0, 255), thickness=3):
    x1, y1 = start
    x2, y2 = end
    
    cx = int((x1 + x2) / 2 - 100)
    cy = int((y1 + y2) / 2 - 100)
    
    t_values = np.linspace(0, 1, 40)
    points = []
    for t in t_values:
        xt = int((1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2)
        yt = int((1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2)
        points.append((xt, yt))
        
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    
    if len(points) >= 3:
        p_last = points[-1]
        p_prev = points[-3]
        cv2.arrowedLine(img, p_prev, p_last, color, thickness=thickness, tipLength=0.5, line_type=cv2.LINE_AA)

class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.current_emotion = "Neutral"
        self.locked_emotion = "Neutral"
        self.lock_until = 0.0
        self.ready_for_next = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_raw = frame.to_ndarray(format="bgr24")
        img_raw = cv2.flip(img_raw, 1)
        h, w, _ = img_raw.shape
        
        main_img = img_raw.copy()
        clean_img = img_raw.copy()
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
        results = self.detector.detect(mp_image)

        emocion_detectada = "Neutral"
        face_box = None

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                ear_izq = calcular_ear(face_landmarks, OJO_IZQ_SUP, OJO_IZQ_INF, OJO_IZQ_EXT, OJO_IZQ_INT, w, h)
                ear_der = calcular_ear(face_landmarks, OJO_DER_SUP, OJO_DER_INF, OJO_DER_EXT, OJO_DER_INT, w, h)
                ear_val = (ear_izq + ear_der) / 2.0

                mar_val = calcular_mar(face_landmarks, w, h)

                comisura_izq_y = face_landmarks[COMISURA_IZQ].y
                comisura_der_y = face_landmarks[COMISURA_DER].y
                centro_labios_y = (face_landmarks[LABIO_SUP].y + face_landmarks[LABIO_INF].y) / 2.0
                altura_comisuras = centro_labios_y - ((comisura_izq_y + comisura_der_y) / 2.0)

                emocion_detectada = clasificar_emocion(ear_val, mar_val, altura_comisuras)

                xs = [lm.x * w for lm in face_landmarks]
                ys = [lm.y * h for lm in face_landmarks]
                xmin, xmax = int(min(xs)), int(max(xs))
                ymin, ymax = int(min(ys)), int(max(ys))
                face_box = (xmin, ymin, xmax, ymax)

                # Dibujar malla de puntos (grafos) en primer plano (main_img) con alta calidad (radio adaptativo)
                circle_radius = max(2, int(w / 600))
                for lm in face_landmarks:
                    cv2.circle(main_img, (int(lm.x * w), int(lm.y * h)), circle_radius, (0, 255, 0), -1, cv2.LINE_AA)

        now = time.time()
        # Mantener la emoción fija 1 segundo cuando se detecta
        if now < self.lock_until:
            emocion_actual = self.locked_emotion
        else:
            if emocion_detectada != "Neutral":
                self.locked_emotion = emocion_detectada
                self.lock_until = now + 1.0
                emocion_actual = emocion_detectada
            else:
                emocion_actual = "Neutral"
                self.locked_emotion = "Neutral"

        self.current_emotion = emocion_actual

        if emocion_actual != "Neutral" and face_box is not None:
            if emocion_actual in EMOTION_IMAGES:
                emo_icon = EMOTION_IMAGES[emocion_actual]
                icon_h, icon_w, _ = emo_icon.shape
                if 20 + icon_h < h and 20 + icon_w < w:
                    main_img[20:20+icon_h, 20:20+icon_w] = emo_icon
                    cv2.rectangle(main_img, (20, 20), (20 + icon_w, 20 + icon_h), (255, 255, 255), 3)

            xmin, ymin, xmax, ymax = face_box
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            
            axes_x = int((xmax - xmin) / 2 * 1.2)
            axes_y = int((ymax - ymin) / 2 * 1.2)
            
            cv2.ellipse(main_img, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, (0, 0, 255), 3, cv2.LINE_AA)
            
            top_circle_y = center_y - axes_y
            start_point = (160, 160)
            end_point = (center_x, top_circle_y)
            draw_curved_arrow(main_img, start_point, end_point, color=(0, 0, 255), thickness=3)

        pip_w, pip_h = int(w * 0.18), int(h * 0.18)
        pip_resized = cv2.resize(clean_img, (pip_w, pip_h))
        
        x_offset = w - pip_w - 20
        y_offset = 20
        
        cv2.rectangle(main_img, (x_offset - 2, y_offset - 2), (x_offset + pip_w + 2, y_offset + pip_h + 2), (255, 255, 255), 2)
        main_img[y_offset:y_offset+pip_h, x_offset:x_offset+pip_w] = pip_resized
        
        cv2.putText(main_img, "Camara Original", (x_offset, y_offset - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(main_img, format="bgr24")

tab_detector, tab_slides = st.tabs(["Detector en Vivo", "Cómo Funciona"])

with tab_detector:
    # Layout en dos columnas principales
    col_video, col_sidebar = st.columns([2.3, 1], gap="large")

    with col_video:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Vista de Cámara y Detección")
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        webrtc_ctx = webrtc_streamer(
            key="emotion-detector",
            video_processor_factory=EmotionTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280, "max": 1920},
                    "height": {"min": 480, "ideal": 720, "max": 1080}
                },
                "audio": False
            },
            async_processing=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sidebar:
        st.markdown("### Panel de Control")
        emotion_status_placeholder = st.empty()
        
        st.markdown("""
            <div class="glass-card">
                <h4 style="margin-top:0; color:#e5e5e7; font-weight: 700;">Instrucciones</h4>
                <ol style="padding-left: 20px; color: #a1a1a6; font-size: 0.9rem; line-height: 1.6;">
                    <li>Haz clic en <b>Start</b> en el reproductor.</li>
                    <li>Permite el acceso a tu cámara web.</li>
                    <li>Ubícate frente al lente para iniciar el escaneo facial.</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    current_detected = "Neutral"
    if webrtc_ctx.video_processor:
        current_detected = webrtc_ctx.video_processor.current_emotion

    display_text = current_detected

    emotion_status_placeholder.markdown(f"""
        <div class="glass-card" style="text-align: center; margin-bottom: 1.25rem;">
            <span style="color: #86868b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Estado Emocional Actual</span>
            <div class="emotion-pill">
                {display_text}
            </div>
        </div>
    """, unsafe_allow_html=True)

with tab_slides:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: #ffffff; margin-bottom: 0.5rem;">Presentación del Sistema</h2>
            <p style="color: #8e8e93; font-size: 1.1rem;">Arquitectura, funcionamiento y métricas del detector de emociones</p>
        </div>
    """, unsafe_allow_html=True)

    slide_col1, slide_col2, slide_col3, slide_col4 = st.columns(4)
    
    if 'current_slide' not in st.session_state:
        st.session_state.current_slide = 1

    with slide_col1:
        if st.button("01. Introducción", use_container_width=True):
            st.session_state.current_slide = 1
    with slide_col2:
        if st.button("02. Malla Facial", use_container_width=True):
            st.session_state.current_slide = 2
    with slide_col3:
        if st.button("03. Métricas EAR & MAR", use_container_width=True):
            st.session_state.current_slide = 3
    with slide_col4:
        if st.button("04. Motor de Reglas", use_container_width=True):
            st.session_state.current_slide = 4

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.current_slide == 1:
        st.markdown("""
            <div class="glass-card" style="padding: 3rem; border-left: 5px solid #ffffff;">
                <div style="font-size: 0.9rem; font-weight: 700; color: #86868b; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1rem;">Diapositiva 01 / 04</div>
                <h3 style="color: #ffffff; font-size: 2rem; font-weight: 800; margin-top: 0; margin-bottom: 1.5rem;">Introducción al Proyecto</h3>
                <p style="color: #d1d1d6; line-height: 1.8; font-size: 1.15rem; margin-bottom: 2rem;">
                    Sistema inteligente de reconocimiento facial en tiempo real desarrollado con Visión por Computador y Machine Learning para identificar emociones humanas con alta precisión.
                </p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 2rem;">
                    <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-top: 0; font-size: 1.1rem;">Análisis en Vivo</h4>
                        <p style="color: #86868b; font-size: 0.95rem; margin-bottom: 0;">Procesamiento continuo fotograma por fotograma sin retrasos.</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-top: 0; font-size: 1.1rem;">5 Emociones</h4>
                        <p style="color: #86868b; font-size: 0.95rem; margin-bottom: 0;">Clasificación instantánea: Felicidad, Tristeza, Enojo, Sorpresa y Neutral.</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-top: 0; font-size: 1.1rem;">Modo Dual</h4>
                        <p style="color: #86868b; font-size: 0.95rem; margin-bottom: 0;">Vista limpia principal con indicador gráfico y visor PiP secundario.</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    elif st.session_state.current_slide == 2:
        st.markdown("""
            <div class="glass-card" style="padding: 3rem; border-left: 5px solid #ffffff;">
                <div style="font-size: 0.9rem; font-weight: 700; color: #86868b; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1rem;">Diapositiva 02 / 04</div>
                <h3 style="color: #ffffff; font-size: 2rem; font-weight: 800; margin-top: 0; margin-bottom: 1.5rem;">Arquitectura y Malla Facial (MediaPipe)</h3>
                <p style="color: #d1d1d6; line-height: 1.8; font-size: 1.15rem; margin-bottom: 2rem;">
                    El núcleo del sistema utiliza <b>MediaPipe Face Landmarker</b> de Google para mapear con precisión milimétrica la geometría facial humana.
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem;">
                    <div style="background: rgba(255,255,255,0.03); padding: 1.75rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-top: 0; font-size: 1.2rem;">478 Puntos Tridimensionales</h4>
                        <p style="color: #86868b; font-size: 1rem; line-height: 1.6; margin-bottom: 0;">
                            Cada fotograma procesa coordenadas 3D de cejas, ojos, labios y contorno facial, permitiendo un seguimiento robusto ante movimientos de cabeza.
                        </p>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 1.75rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-top: 0; font-size: 1.2rem;">Visor PiP Secundario</h4>
                        <p style="color: #86868b; font-size: 1rem; line-height: 1.6; margin-bottom: 0;">
                            Incorpora una ventana superpuesta en tiempo real que muestra la malla poligonal completa generada por la IA.
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    elif st.session_state.current_slide == 3:
        st.markdown("""
            <div class="glass-card" style="padding: 3rem; border-left: 5px solid #ffffff;">
                <div style="font-size: 0.9rem; font-weight: 700; color: #86868b; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1rem;">Diapositiva 03 / 04</div>
                <h3 style="color: #ffffff; font-size: 2rem; font-weight: 800; margin-top: 0; margin-bottom: 1.5rem;">Métricas Geométricas (EAR y MAR)</h3>
                <p style="color: #d1d1d6; line-height: 1.8; font-size: 1.15rem; margin-bottom: 2rem;">
                    La clasificación se basa en fórmulas matemáticas de proporciones faciales derivadas de los landmarks detectados.
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem;">
                    <div style="background: rgba(255,255,255,0.03); padding: 1.75rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-top: 0; font-size: 1.2rem;">EAR (Eye Aspect Ratio)</h4>
                        <p style="color: #86868b; font-size: 1rem; line-height: 1.6; margin-bottom: 0;">
                            Relación entre la distancia vertical y horizontal de los ojos. Permite detectar estados de sorpresa (ojos muy abiertos) o concentración/enojo.
                        </p>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 1.75rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-top: 0; font-size: 1.2rem;">MAR (Mouth Aspect Ratio)</h4>
                        <p style="color: #86868b; font-size: 1rem; line-height: 1.6; margin-bottom: 0;">
                            Mide la apertura de la boca y la altura de las comisuras labiales respecto al centro para diferenciar sonrisas de gestos tristes.
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="glass-card" style="padding: 3rem; border-left: 5px solid #ffffff;">
                <div style="font-size: 0.9rem; font-weight: 700; color: #86868b; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1rem;">Diapositiva 04 / 04</div>
                <h3 style="color: #ffffff; font-size: 2rem; font-weight: 800; margin-top: 0; margin-bottom: 1.5rem;">Motor de Clasificación y Retención</h3>
                <p style="color: #d1d1d6; line-height: 1.8; font-size: 1.15rem; margin-bottom: 2rem;">
                    El sistema combina las métricas con un motor de reglas heurísticas y un sistema de control temporal.
                </p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 2rem;">
                    <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-top: 0; font-size: 1.1rem;">Reglas Heurísticas</h4>
                        <p style="color: #86868b; font-size: 0.95rem; margin-bottom: 0;">Evaluación lógica instantánea de umbrales en cada fotograma.</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-top: 0; font-size: 1.1rem;">Retención de 1s</h4>
                        <p style="color: #86868b; font-size: 0.95rem; margin-bottom: 0;">Fija cada emoción detectada durante 1 segundo para evitar fluctuaciones.</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-top: 0; font-size: 1.1rem;">Feedback Visual</h4>
                        <p style="color: #86868b; font-size: 0.95rem; margin-bottom: 0;">Óvalo de enfoque, flecha guía e imagen representativa en pantalla.</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

time.sleep(1)
st.rerun()
