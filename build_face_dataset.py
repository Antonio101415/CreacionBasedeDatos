# Heimdall-EYE USO:
# python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/Antonio ( O nombre de la base de datos)

# Importamos los paquetes necesarios o librerias
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

# Contruimos el analizador de argumentos y analizamos este argumento
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
args = vars(ap.parse_args())

# Cargamos la cascada Haar de OpenCV para la deteccion de rostros 
detector = cv2.CascadeClassifier(args["cascade"])

# Inicializamos la transmision de video que permite que la camara se encienda
# He inicializamos el numero total de caras encontradas hasta ahora
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

# Recorremos los fotogramas de la retrasmision del video
while True:
	# Agarra el cuadro de la secuencia de video enhebrada 
	#En caso de que queramos inscribirlo en el disco y luego cambiar el tamaño del marco
	# Para que podamos aplicar mas rapido la deteccion de rostros
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# Detectamos rostros en una escala de grises
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

	# Bucle sobre las detecciones de la cara y dibujarlas en el marco 
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Vemos el frame de salida
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# Cuando queramos realizar la foto solo tenemos que pulsar la tecla K 
	# Automaticamente se guardar en la base de datos , en la carpeta especificada en la ruta 
	if key == ord("k"):
		p = os.path.sep.join([args["output"], "{}.png".format(
			str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1

	# Cortamos el bucle con la letra Q cuando queramos dejar de obtener fotos
	elif key == ord("q"):
		break


print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
