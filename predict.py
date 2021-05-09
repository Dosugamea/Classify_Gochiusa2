import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import cv2
import os.path


class Predictor(object):
    MODEL_FILE = "./local_train.h5"
    LABELS = [
        'Blue Mountain', 'Chino', 'Chiya',
        'Cocoa', 'Maya', 'Megumi',
        'Mocha', 'Rize', 'Sharo'
    ]
    OUTPUT_DIR = "faces"
    INPUT_SIZE = 224
    model = None
    face_cascade = None

    def __init__(self):
        self.model = load_model(self.MODEL_FILE)
        self.face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')

    def load_image(self, path):
        img = cv2.imread(path)
        return img

    def detect_face(self, img):
        faces = self.face_cascade.detectMultiScale(img, minNeighbors=10)
        imgs = []
        for i, (x, y, w, h) in enumerate(faces):
            face_image = img[y:y+h, x:x+w]
            output_path = os.path.join(self.OUTPUT_DIR, '{0}.jpg'.format(i))
            cv2.imwrite(output_path, face_image)
            print(output_path)
            temp_img = load_img(output_path, target_size=(self.INPUT_SIZE, self.INPUT_SIZE))
            temp_img_array = img_to_array(temp_img)
            temp_img_array = temp_img_array.astype('float32') / 255.0
            temp_img_array = temp_img_array.reshape((self.INPUT_SIZE, self.INPUT_SIZE, 3))
            imgs.append(temp_img_array)
        return imgs

    def predict_single(self, filename):
        temp_img = load_img(filename, target_size=(self.INPUT_SIZE, self.INPUT_SIZE))
        temp_img_array = img_to_array(temp_img)
        temp_img_array = temp_img_array /255.0
        temp_img_array = temp_img_array.reshape((self.INPUT_SIZE, self.INPUT_SIZE, 3))
        prd = self.model.predict(np.array([temp_img_array]))
        print(self.LABELS)
        print([round(v, 4) for v in prd[0]])
        prelabel = np.argmax(prd)
        return self.LABELS[int(prelabel)]

    def predict(self, filename):
        img = self.load_image(filename)
        face_imgs = self.detect_face(img)
        labels = []
        for i, img in enumerate(face_imgs):
            prd = self.model.predict(np.array([img]))
            # 精度の表示
            print(self.LABELS)
            print([round(v, 4) for v in prd[0]])
            prelabel = np.argmax(prd)
            labels.append(self.LABELS[int(prelabel)])
        return labels


if __name__ == '__main__':
    filename = "./faces/2.jpg"
    cl = Predictor()
    print(cl.predict_single(filename))
