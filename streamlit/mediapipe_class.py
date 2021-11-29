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
import time
from tqdm import tqdm
import tempfile
from pathlib import Path
import base64
import glob


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

st.set_option('deprecation.showfileUploaderEncoding', False)
filepathdesu='/Users/shinohararikunin/Desktop/model'
modeldesu = tf.keras.models.load_model(filepathdesu+'/Emotions_model.hdf5')

classes=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
@st.cache(allow_output_mutation=True)
def get_squre_image(target_img):
    '''画像に余白を加えて正方形にする'''
    bg_color = target_img.resize((1, 1)).getpixel((0, 0))  # 余白は全体の平均色
    width, height = target_img.size
    if width == height:
	    return target_img
    elif width > height:
	    resized_img = Image.new(target_img.mode, (width, width), bg_color)
	    resized_img.paste(target_img, (0, (width - height) // 2))
	    return resized_img
    else:
	    resized_img = Image.new(target_img.mode, (height, height), bg_color)
	    resized_img.paste(target_img, ((height - width) // 2, 0))
	    return resized_img

def mediapipe_static(dir_input):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
    a=1
    #img=cv2.imread(dir_input)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        img=cv2.imread(dir_input)
        #results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # mediapipeで推定可能な画像に対して、3次元カラムと推定3次元座標をlistに追加
        
        #annotated_image = img.copy()
        """
        for face_landmarks in results.multi_face_landmarks:
            facemesh_csv = []
            col_label = []

            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
                    
            for xyz, landmark in enumerate(face_landmarks.landmark):
                col_label.append(str(xyz) + "_x")
                col_label.append(str(xyz) + "_y")
                col_label.append(str(xyz) + "_z")
                facemesh_csv.append(landmark.x)
                facemesh_csv.append(landmark.y)
                facemesh_csv.append(landmark.z)
        data = pd.DataFrame([facemesh_csv], columns=col_label)
        #image=cv2.imwrite(annotated_image)
    #return data
    """
    return img



def main():
    st.title('感情分類器')
    st.write("NNモデルを用いて、アップロードした画像をmediapipeから分類する")
    uploaded_file = st.file_uploader('Choose a image file to predict')
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            fp=Path(tmp_file.name)
            
        image_pil_array=Image.open(uploaded_file)
        st.image(
            image_pil_array,caption='uploaded image',
            use_column_width=True
        )
        fp="/Users/shinohararikunin/Desktop/privatemysite/image/gakki.jpeg"
        image = Image.open(fp)
        st.image(image, caption='サンプル',use_column_width=True)
        

        pd_img=mediapipe_static(fp)
        st.image(pd_img, caption='サンプル',use_column_width=True)
        #st.dataframe(pd_img)
        st.write(pd_img)

if __name__=='__main__':
    main()


