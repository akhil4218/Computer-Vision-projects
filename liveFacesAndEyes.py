import cv2 
import numpy as np

cascade_face = 'cascades/haarcascade_frontalface_default.xml'

face_classifier = cv2.CascadeClassifier(cascade_face)

def face_and_eye_detector(image):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faces = face_classifier.detectMultiScale(gray, 1.2, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 3)

		area_gray = gray[y:y+h, x:x+w]
		area_original = image[y:y+h, x:x+w]
		
	image = cv2.flip(image, 1)		
	return image				

capture = cv2.VideoCapture(0)

while True:
	response, frame = capture.read()
	cv2.imshow("Live Face Classifier", face_and_eye_detector(frame))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()
