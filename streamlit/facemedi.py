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
        max_num_faces=max_num_faces,
        min_detection_confidence=min_detection_confidence).process(cv2.cvtColor(np.array(dir_input), cv2.COLOR_BGR2RGB))

    annotated_image = np.array(dir_input)
        
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    return annotated_image

st.write(
    """ # Mediapipeから顔特徴点を表示します!!"""
)
st.sidebar.info("設定")
max_num_faces = st.sidebar.slider('max num faces:', 1, 4, 2, 1)
min_detection_confidence = st.sidebar.slider('Minimum detection confidence:', 0.1, 1.0, 0.5, 0.1)
st.sidebar.info("顔入力")
uploaded_files = st.sidebar.file_uploader('Choose a image file to predict')
if uploaded_files is not None:
    image_pil_array = Image.open(uploaded_files)
    st.image(
        image_pil_array, caption='uploaded image',
        use_column_width=True
    )
    track_button = st.sidebar.button('Track Faces')
    if track_button:
        annotated_images = mediapipe_static(image_pil_array, max_num_faces, min_detection_confidence)
        st.image(annotated_images)