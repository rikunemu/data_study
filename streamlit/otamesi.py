import cv2
import time
import mediapipe as mp
import pandas as pd
import streamlit as st
import tensorflow as tf
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
desu='/Users/shinohararikunin/Desktop/model'
modeldesu = tf.keras.models.load_model(desu+'/lstmpose.hdf5')

def medi_image(image_pil_array:'PIL.Image',results):

    if not results.face_landmarks and not results.left_hand_landmarks and not results.right_hand_landmarks and not results.pose_landmarks:
        a=1
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
    data = pd.DataFrame([holmesh_csv], columns=col_label)
    time.sleep(1)

    return data

def pre_image(data):
    target=[]
    target.append(data.iloc[0:30,:])
    x_target = np.array(target)
    x_test = x_target.astype(np.float)


st.title('ポーズ分類')
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
    results = holistic.process(image)
    #st.write(1)
    df=medi_image(image,results)
    st.write(df)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
