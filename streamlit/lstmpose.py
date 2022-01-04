import cv2
#import time
import mediapipe as mp
import pandas as pd
import streamlit as st
#import keras
from keras import models
from keras.models import Sequential
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
desu='/Users/shinohararikunin/Desktop/model'
modeldesu = models.load_model(desu+'/lstmpose2.h5')
idx=0
data=[]
def medi_image(image:'PIL.Image',results,idx,data):
    

    if not results.face_landmarks and not results.left_hand_landmarks and not results.right_hand_landmarks and not results.pose_landmarks:
        return
    # mediapipeで推定可能な画像に対して、3次元カラムと推定3次元座標をlistに追加
    else:
        annotated_image = image.copy()
        holmesh_csv = []
        col_label = []

        mp_drawing.draw_landmarks(
            annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        if results.right_hand_landmarks:
            for xyz, right_hand_landmark in enumerate(results.right_hand_landmarks.landmark):
                col_label.append("right_hand_"+str(xyz) + "_x")
                col_label.append("right_hand_"+str(xyz) + "_y")
                col_label.append("right_hand_"+str(xyz) + "_z")
                holmesh_csv.append(right_hand_landmark.x)
                holmesh_csv.append(right_hand_landmark.y)
                holmesh_csv.append(right_hand_landmark.z)
        if not results.right_hand_landmarks:
            for xyz in range(21):
                col_label.append("right_hand_"+str(xyz) + "_x")
                col_label.append("right_hand_"+str(xyz) + "_y")
                col_label.append("right_hand_"+str(xyz) + "_z")
                for _ in range(3):
                    holmesh_csv.append(0.1)
        if results.pose_landmarks:
            for xyz, pose_landmark in enumerate(results.pose_landmarks.landmark):
                col_label.append("pose_"+str(xyz) + "_x")
                col_label.append("pose_"+str(xyz) + "_y")
                col_label.append("pose_"+str(xyz) + "_z")
                holmesh_csv.append(pose_landmark.x)
                holmesh_csv.append(pose_landmark.y)
                holmesh_csv.append(pose_landmark.z)
        if not results.pose_landmarks:
            for xyz in range(33):
                col_label.append("pose_"+str(xyz) + "_x")
                col_label.append("pose_"+str(xyz) + "_y")
                col_label.append("pose_"+str(xyz) + "_z")
                for _ in range(3):
                    holmesh_csv.append(0.1)
    
    # 1枚目の画像をDataFrame構造で保存
    if idx == 0:
        data = pd.DataFrame([holmesh_csv], columns=col_label)
        idx+=1
    # 1枚目と2枚目以降のDataFrameを縦に結合
    else:
        data1 = pd.DataFrame([holmesh_csv], columns=col_label)
        data = pd.concat([data, data1], ignore_index=True)
        idx+=1
    #time.sleep(1)

    return data,idx

def pre_image(data):
    target=[]
    target.append(data.iloc[0:30,:])
    target = np.array(target).reshape(1, 30,162)
    x_target = np.array(target)
    x_test = x_target.astype(np.float)
    #pred=modeldesu.predict(x_test)
    pred = np.argmax(modeldesu.predict(x_test),axis=-1)

    return pred


st.title('ポーズ分類')
image_loc = st.empty()
scores_st = st.empty()
label_st = st.empty()
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_loc.image(image)

    results= holistic.process(image)
    #st.write(1)
    data,idx=medi_image(image,results,idx,data)
    label_st.text(idx)
    if idx==30:
        #st.write(data)
        kekka=pre_image(data)
        idx=0
        data=0
        #kekka=kekka+1
        scores_st.text(kekka)
        
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
