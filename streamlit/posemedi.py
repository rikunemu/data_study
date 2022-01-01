import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import time
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

st.set_option('deprecation.showfileUploaderEncoding', False)
desu='/Users/shinohararikunin/Desktop/model'
modeldesu = tf.keras.models.load_model(desu+'/lstmpose.h5')
faceclasses=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@st.cache(allow_output_mutation=True)
def get_square_image(target_img):
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

def medi_image(image_pil_array:'PIL.Image'):
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:

        image = cv2.imread(image_pil_array)
      #image_height, image_width, _ = image.shape
      # 画像の色の変換
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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

def unko(image):
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        a = 0
    return a


st.title('ポーズ分類')

st.write("lstm modelを使って、リアルタイム動画を分類します。")
# Webカメラ入力の場合：
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        #st.write(1)
        success, image = cap.read()
        if not success:
            #print("Ignoring empty camera frame.")
            st.write(1)
        # ビデオをロードする場合は、「continue」ではなく「break」を使用してください
            continue

        # 後で自分撮りビューを表示するために画像を水平方向に反転し、BGR画像をRGBに変換
        image.flags.writeable = False
        st.write(1)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # パフォーマンスを向上させるには、オプションで、参照渡しのためにイメージを書き込み不可としてマーク
        st.write(1)
        results = holistic.process(image)
        st.write(1)

        # 画像にランドマークアノテーションを描画
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        #st.image(image)
