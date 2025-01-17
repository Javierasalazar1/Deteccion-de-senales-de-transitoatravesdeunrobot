import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
from flask import Flask, Response
from threading import Thread
from picamera2 import Picamera2
from picarx import Picarx  # Importar librerÃ­a de control del robot

# Configuraciones del modelo
TFLITE_MODEL_PATH = '/home/py3/Desktop/TESIS/modelo.tflite'
IMAGE_SIZE = (224, 224)
CLASSES = ['0', '1', '2', '3', '4']  # 0: Pare, 1: Avanzar, 2: Izquierda, 3: Derecha, 4: Fondo

# ConfiguraciÃ³n de Flask
app = Flask(__name__)
frame_to_stream = None  # Frame que se enviar al navegador


class TrafficSignDetector:
    def __init__(self, model_path):
        # Cargar modelo TFLite
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Detalles de entrada y salida
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Inicializar cÃ¡mara
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration()
        self.picam2.configure(config)
        self.picam2.start()
        
        # Inicializar robot Picarx
        self.px = Picarx()
        
        # Variables para resultados
        self.current_sign = None
        self.confidence = 0
        self.detection_thread = None
        self.running = False
        
        # Nuevas variables de seguimiento
        self.last_sign = None
        self.last_sign_timestamp = 0
        self.no_sign_duration = 0
        self.after_turn_timestamp = 0

    def preprocess_image(self, frame):
        """Preprocesa la imagen para el modelo"""
        # Redimensionar
        resized = cv2.resize(frame, IMAGE_SIZE)
        
        # Convertir a formato float32 y normalizar
        image_float = resized.astype(np.float32)
        image_float = (image_float - 127.5) / 127.5
        
        # Añadir dimension de batch
        input_data = np.expand_dims(image_float, axis=0)
        
        return input_data

    def detect_sign(self):
        """Detecta seÃ±ales de trÃ¡fico continuamente"""
        global frame_to_stream

        while self.running:
            # Capturar frame
            frame = self.picam2.capture_array()
            
            # Reducir resoluciÃ³n del frame para transmision
            frame = cv2.resize(frame, (800, 600))  
           
            # Convertir de RGB a BGR para OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Preprocesar imagen para inferencia
            input_data = self.preprocess_image(frame)
            
            # Establecer tensor de entrada
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Ejecutar inferencia
            self.interpreter.invoke()
            
            # Obtener predicciones
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Obtener la clase con mayor probabilidad
            prediction = np.argmax(output_data[0])
            confidence = np.max(output_data[0])
            
            # Verificar si estamos en periodo de avance después de un giro
            if (self.after_turn_timestamp > 0 and 
                time.time() - self.after_turn_timestamp < 3):
                # Seguir avanzando por 3 segundos después del giro
                self.px.forward(50)
                
                # Dibujar texto en el frame para indicar avance post-giro
                cv2.putText(frame_bgr, "Post-turn advance", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Actualizar resultados si la confianza es alta
            if confidence > 0.7:  
                self.current_sign = CLASSES[prediction]
                self.confidence = confidence * 100

                # Resetear la duración sin señal
                self.no_sign_duration = 0

                # Logica para controlar el movimiento del robot
                if self.current_sign in ['2', '3']:  # Si es giro izquierda o derecha
                    # Marcar timestamp de inicio de avance post-giro
                    self.after_turn_timestamp = time.time()

                self.control_robot(self.current_sign)
                
                # Guardar la última señal detectada
                if self.current_sign != '4':
                    self.last_sign = self.current_sign
                    self.last_sign_timestamp = time.time()
            else:
                # Incrementar la duración sin señal
                self.no_sign_duration += 1

                # Detener avance post-giro si pasa más de 3 segundos
                if (self.after_turn_timestamp > 0 and 
                    time.time() - self.after_turn_timestamp >= 3):
                    self.px.stop()
                    self.after_turn_timestamp = 0

                # Si no hay señal después de ver una señal previamente
                if (self.last_sign is not None and 
                    time.time() - self.last_sign_timestamp > 1 and 
                    self.current_sign == '4'):
                    
                    # Avanzar 1 segundo después de última señal
                    self.px.forward(50)
                    time.sleep(1)
                    self.px.stop()
                    
                    # Reiniciar variables de seguimiento
                    self.last_sign = None
                    self.last_sign_timestamp = 0
            
            # Dibujar texto en el frame
            cv2.putText(frame_bgr, f"Sign: {self.current_sign}, Conf: {self.confidence:.2f}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Codificar el frame para transmision
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Calidad media para reducir carga
            frame_to_stream = cv2.imencode('.jpg', frame_bgr, encode_param)[1].tobytes()

    def control_robot(self, current_sign):
        """Controla el robot segÃºn la seÃ±al detectada"""
        if current_sign == '0':  # Clase 0: Pare
            self.px.stop()
            print("Clase detectada: Detener")
        elif current_sign == '1':  # Clase 1: Avanzar
            self.px.forward(50)  # Velocidad del motor: 50
            print("Clase detectada: Avanzar")
        elif current_sign == '2':  # Clase 2: Izquierda
            self.px.set_dir_servo_angle(30)  # Corrige el angulo para girar a la izquierda
            self.px.forward(50)
            time.sleep(1.3)  # Giro de 1 segundo
            self.px.set_dir_servo_angle(0)  # Endereza las ruedas
            print("Clase detectada: Girar a la izquierda")
        elif current_sign == '3':  # Clase 3: Derecha
            self.px.set_dir_servo_angle(-30)  # Corrige el angulo para girar a la derecha
            self.px.forward(50)
            time.sleep(1.3)  # Giro de 1 segundo
            self.px.set_dir_servo_angle(0)  # Endereza las ruedas
            print("Clase detectada: Girar a la derecha")
        elif current_sign == '4':  # Clase 4: Fondo
            self.px.stop()
            print("Clase detectada: Fondo (sin accion)")

    def start_detection(self):
        """Inicia el hilo de deteccion"""
        self.running = True
        self.detection_thread = Thread(target=self.detect_sign)
        self.detection_thread.start()

    def stop_detection(self):
        """Detiene el hilo de deteccion"""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join()
        self.picam2.stop()


@app.route('/video_feed')
def video_feed():
    """Ruta para la transmision de video"""
    def generate():
        global frame_to_stream
        while True:
            if frame_to_stream:
                # Crear flujo MJPEG
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_to_stream + b'\r\n')
            else:
                time.sleep(0.05)  # Esperar brevemente si no hay frames listos
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    # Crear detector
    detector = TrafficSignDetector(TFLITE_MODEL_PATH)
    
    try:
        # Iniciar deteccion
        detector.start_detection()
        
        # Iniciar Flask en un hilo
        flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=9000, debug=False, use_reloader=False))
        flask_thread.start()
        
        # Bucle principal
        while True:
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nDeteniendo deteccion...")
    finally:
        # Detener detector
        detector.stop_detection()

if __name__ == "__main__":
    main()
