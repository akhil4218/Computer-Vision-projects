import cv2
import numpy as np

def canny(frame):
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
	edges = cv2.Canny(gray_blur, 10, 70)
	ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
	return mask


capture = cv2.VideoCapture(0)

while (True):
	response, frame = capture.read()
	cv2.imshow("Normal Image",frame)
	cv2.imshow("Canny edges", canny(frame))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()			
