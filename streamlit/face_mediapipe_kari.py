import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps , ImageDraw, ImageFont
import cv2
import io
import os
import json
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles

#classes=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def track_faces_in_images(dir_input,max_num_faces,min_detection_confidence):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    results = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5).process(cv2.cvtColor(np.array(dir_input), cv2.COLOR_BGR2RGB))

    # mediapipeで推定不可の画像に対して、3次元カラムとNaN要素をlistに追加
    if not results.multi_face_landmarks:
        facemesh_csv = []
        col_label = []
        for xyz in range(468):
            col_label.append(str(xyz) + "_x")
            col_label.append(str(xyz) + "_y")
            col_label.append(str(xyz) + "_z")
        for _ in range(3):
            facemesh_csv.append(np.nan)
    else:
        annotated_image = cv2.cvtColor(np.array(dir_input.copy()), cv2.COLOR_BGR2RGB)
        
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        
        facemesh_csv = []
        col_label = []
        
        for xyz, landmark in enumerate(face_landmarks.landmark):
            col_label.append(str(xyz) + "_x")
            col_label.append(str(xyz) + "_y")
            col_label.append(str(xyz) + "_z")
            facemesh_csv.append(landmark.x)
            facemesh_csv.append(landmark.y)
            facemesh_csv.append(landmark.z)
    
    data = pd.DataFrame([facemesh_csv], columns=col_label)

    return data, annotated_image

st.title('画像分類器')

st.write("cnn modelを使って、アップロードした画像を分類します。")
image_pil_array=0
uploaded_file = st.file_uploader('Choose a image file to predict')

if uploaded_file is not None:
    image_pil_array = Image.open(uploaded_file)
    st.image(
        image_pil_array, caption='uploaded image',
        use_column_width=True
    )
    an_image=track_faces_in_images(image_pil_array,1,0.1)
    st.image(an_image,caption='uploaded image',use_column_width=True)