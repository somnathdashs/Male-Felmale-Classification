import cv2,numpy as np
import tensorflow as tf

# "gender.h5" -> (1,60,60,3)
# "gender2.h5" -> (1,128,128,3)


def get_Model():
    return tf.keras.models.load_model("./Gender.h5")



def ISFace(img):
    face = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
    gre = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face123 = face.detectMultiScale(gre,1.1,4)
    # print(face123)
    return True if len(face123)>0 else False

M=get_Model()
Class=["Female",'Male']  
im=[]

cam=cv2.VideoCapture(0)
while True:
    _,img=cam.read()
    img=cv2.flip(img,180)
    if ISFace(img):
        face = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
        gre = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face123 = face.detectMultiScale(gre,1.1,4)
        # vid = cv2.VideoCapture(0)
        for(x,y,w,h) in face123 :
            cimage=img[y:y+h,x:x+w]
            simage=img[y:y+h,x:x+w]
            simage=cv2.resize(simage,(500,500))
            cimage=cv2.resize(cimage,(60,60))
            i=np.array(cimage)/255
            i=i.reshape((1,60,60,3))
            cimage=i
            pr=M.predict([cimage])
            i=np.argmax(pr)
            cas=Class[i]
            pr=int(pr[0][i]*100)
            if pr>80:
                cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
            cv2.putText(img,cas+" "+str(pr)+"%",(x+w,y+h),5,1,(0,255,255),2)
            cv2.putText(simage,cas+" "+str(pr)+"%",(0,0+50),5,1,(0,255,255),2)
            # print(tf.estimator.evaluate())
        cv2.imshow("CImage",simage)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    



