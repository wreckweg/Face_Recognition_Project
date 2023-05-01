import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img1 = cv2.imread('test2.jpg', 0)

def face_extractor(img):
    # detects face and returns cropped images

    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # crop
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + h + 50]
    return cropped_face


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1

        face = cv2.resize(face_extractor(frame), (400, 400))
        file_name_path = './images/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Cropper', face)

    else:
        print('Face_not_found')
        pass
    if cv2.waitKey(1) == 13 or count == 100:  # 13 is enter
        break
cap.release()
cv2.destroyAllWindows()
