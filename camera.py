import cv2
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
#face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ds_factor=0.6
# def prepImg(pth):
#     return cv2.resize(pth,(224,224)).reshape(1,224,224,3)/255.0

class VideoCamera(object):
    def __init__(self):
        print("hello")
        self.video = cv2.VideoCapture(0)
        with open('model.json', 'r') as f:
            loaded_model_json = f.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("model.h5")
        print("Loaded model from disk")

        self.resMap = {
                0 : 'Mask On',
                1 : 'Kindly Wear Mask'
            }

        self.colorMap = {
                0 : (0,255,0),
                1 : (0,0,255)
            }

    
    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        # success, image = self.video.read()
        # image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        # gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        # for (x,y,w,h) in face_rects:
        # 	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        # 	break
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        ret,img = self.video.read()
        faces = classifier.detectMultiScale(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),1.1,2)
        for face in faces:
            slicedImg = img[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
            pred = self.model.predict(cv2.resize(img,(224,224)).reshape(1,224,224,3)/255.0) #
            pred = np.argmax(pred)
            cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),self.colorMap[pred],2)
            cv2.putText(img, self.resMap[pred],(face[0],face[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)        
            
        re, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()           
        #cv2.imshow('FaceMask Detection',img)

            
