import cv2 as cv
import face_recognition as fr
import numpy as np
from tkinter import filedialog

file_1= filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select first File", 
                                          filetypes = (("Text files", 
                                                        "*.*"), 
                                                       ("all files", 
                                                        "*.*")))
file_2= filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select second File", 
                                          filetypes = (("Text files", 
                                                        "*.*"), 
                                                       ("all files",  
                                                        "*.*")))
        
img_1 = fr.load_image_file(file_1)
img_2 = fr.load_image_file(file_2)

imgcvt_1 = cv.cvtColor(img_1,cv.COLOR_BGR2RGB)
imgcvt_2 = cv.cvtColor(img_2,cv.COLOR_BGR2RGB)

faceLoc_1 = fr.face_locations(imgcvt_1)[0]
faceLoc_2 = fr.face_locations(imgcvt_2)[0]

cv.rectangle(imgcvt_1,(faceLoc_1[3],faceLoc_1[0]),(faceLoc_1[1],faceLoc_1[2]),2)
cv.rectangle(imgcvt_2,(faceLoc_2[3],faceLoc_2[0]),(faceLoc_2[1],faceLoc_2[2]),2)

encoding_1 = fr.face_encodings(imgcvt_1)[0]
encoding_2 = fr.face_encodings(imgcvt_2)[0]

print(fr.compare_faces([encoding_1],encoding_2))
print(fr.face_distance([encoding_1],encoding_2))

cv.imshow("First image",imgcvt_1)
cv.imshow("Second image",imgcvt_2)
cv.waitKey(0)
