import cv2
import tensorflow as tf
import os
import numpy as np
import pickle
import keras
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder

def change_model(model, new_input_shape, custom_objects=None):
    # replace input shape of first layer
    model.layers[0]._batch_input_shape = new_input_shape

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json(),custom_objects=custom_objects)

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            print("Loaded layer {}".format(layer.name))
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model

def implement_gan(img):
    model = keras.models.load_model('generator.h5')
    model = change_model(model, new_input_shape=[None, None, None, 3])
    inputs = keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)
    low_res = img

    low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

    # Rescale to 0-1.
    low_res = low_res / 255.0

    # Get super resolution image
    sr = model.predict(np.expand_dims(low_res, axis=0))[0]

    # Rescale values in range 0-255
    sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

    # Convert back to BGR for opencv
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

    return sr




facenet = FaceNet()

face_classs = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = pickle.load(open('model_svm_own', 'rb'))

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classs.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        implement_gan(img)
        img = cv2.resize(img,(160,160))
        #img = img.astype('float32')
        img = np.expand_dims(img,axis=0)
        yPred = facenet.embeddings(img)
        name_face = model.predict(yPred)
        confidence = model.decision_function(yPred)
        max_confi = abs(np.max(confidence[0]))
        print(max_confi)

        percent_ = (max_confi/12)
        formated_max_confi = format(percent_, ".2f")

        fin_name = str(name_face)+" "+ str(formated_max_confi)
        #unknown = "unknown" + str(max_confi)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

        if max_confi >11.8:
            cv2.putText(frame, fin_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Unknown" , (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)


    cv2.imshow('Face recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):  # FF to provide mask for 64 bit machine
        break


cap.release()
cv2.destroyAllWindows()


