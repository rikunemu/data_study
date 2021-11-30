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

def track_faces_in_images(file_list,max_num_faces,min_detection_confidence):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
    annotated_image=[]
    multi_faceednesses=[]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence) as face_mesh:
        for upload_file in file_list:
            #image=cv2.imdecode(np.frombuffer(upload_file.read(),np.uint8),1)
            #image=cv2.flip(image,1)
            image=np.array(upload_file)
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            #multi_faceednesses.append(results.multi_faceednesses)
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            annotated_image.append(image)
        
    return annotated_image,multi_faceednesses

st.write(
    """ # Face Tracking in Pictures Using Google's MediaPipe Hands  """
)
st.sidebar.info("General settings")
max_num_faces = st.sidebar.slider('Max number of faces:', 1, 4, 2, 1)
min_detection_confidence = st.sidebar.slider('Minimum detection confidence:', 0.1, 1.0, 0.5, 0.1)
st.sidebar.info("Input")
uploaded_files = st.sidebar.file_uploader("Upload JPG images of faces:", type=['jpg'], accept_multiple_files=True)
if len(uploaded_files) > 0:
    show_results = st.sidebar.checkbox('Show classification results')
    track_button = st.sidebar.button('Track Faces')
    if track_button:
        annotated_images, multi_faceednesses = track_faces_in_images(uploaded_files, max_num_faces, min_detection_confidence)
        if len(annotated_images) > 0:
            for idx, annotated_image in enumerate(annotated_images):
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(cv2.flip(annotated_image, 1))
                if show_results:
                    st.write(str(multi_faceednesses[idx]))