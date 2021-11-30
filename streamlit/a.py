# -*- coding: utf-8 -*-
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import streamlit as st
from PIL import Image, ImageOps , ImageDraw, ImageFont

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def mediapipe_static(dir_input,max_num_faces,min_detection_confidence):
    

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
        #annotated_image = cv2.cvtColor(np.array(dir_input.copy()), cv2.COLOR_BGR2RGB)
        annotated_image = np.array(dir_input)
        
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

st.write(
    """ # Face Tracking in Pictures Using Google's MediaPipe Hands  """
)
st.sidebar.info("General settings")
max_num_faces = st.sidebar.slider('Max number of faces:', 1, 4, 2, 1)
min_detection_confidence = st.sidebar.slider('Minimum detection confidence:', 0.1, 1.0, 0.5, 0.1)
st.sidebar.info("Input")
uploaded_files = st.sidebar.file_uploader('Choose a image file to predict')
if uploaded_files is not None:
    image_pil_array = Image.open(uploaded_files)
    st.image(
        image_pil_array, caption='uploaded image',
        use_column_width=True
    )
    show_results = st.sidebar.checkbox('Show classification results')
    track_button = st.sidebar.button('Track Faces')
    if track_button:
        multi_faceednesses,annotated_images = mediapipe_static(image_pil_array, max_num_faces, min_detection_confidence)
        st.image(annotated_images)