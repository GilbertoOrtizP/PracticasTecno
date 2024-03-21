import cv2
import face_recognition

img=cv2.imread("ImagenesRostros/Brad Pitt/Imagen1.jpg")
image = cv2.resize(img, (300,450))

face_loc=face_recognition.face_locations(image)
vector_rostro = face_recognition.face_encodings(image, known_face_locations=[face_loc][0])
vector_rostro=vector_rostro[0]
print(vector_rostro)

cv2.rectangle(image, (face_loc[0][3], face_loc[0][0]), (face_loc[0][1], face_loc[0][2]),(255,0,0),5)
cv2.imshow("Imagen",image)
cv2.waitKey(0)
cv2.destroyAllWindows()