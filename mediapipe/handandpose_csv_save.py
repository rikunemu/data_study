# -*- coding: utf-8 -*-


import glob
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

dir_input=r"D:\output_file\image\Normal"

dir_output_image=r"D:\output_file\image\Mediapipe"

dir_output_csv=r"D:\output_file\csvfile"

flag=0

len_sign=20
list_class=["test","train"]


def hol_mediapipe_static(dir_input, dir_output_image, dir_output_csv):
  
  files = glob.glob(dir_input)
  files.sort()
  os.mkdir(dir_output_image)
  dir_output_image=dir_output_image+"/"
  
  with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True) as holistic:
    
    for idx, file in tqdm(enumerate(files, start=0)):
      # 画像の読み込み
      image = cv2.imread(file)
      #image_height, image_width, _ = image.shape
      # 画像の色の変換
      results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      
      # mediapipeで推定不可の画像に対して、3次元カラムとNaN要素をlistに追加
      #if not results.multi_hand_landmarks:
      if not results.face_landmarks and not results.left_hand_landmarks and not results.right_hand_landmarks and not results.pose_landmarks:
        continue
      # mediapipeで推定可能な画像に対して、3次元カラムと推定3次元座標をlistに追加
      else:
        annotated_image = image.copy()
        #condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        #bg_image = np.zeros(image.shape, dtype=np.uint8)
        #bg_image[:] = BG_COLOR
        #annotated_image = np.where(condition, annotated_image, bg_image)
        flag=1
        
        #for hol_landmarks in results.pose_landmarks:
        holmesh_csv = []
        col_label = []

        mp_drawing.draw_landmarks(
            annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(
            annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.face_landmarks:
            flag=1
            for xyz, face_landmark in enumerate(results.face_landmarks.landmark):
                col_label.append("face_"+str(xyz) + "_x")
                col_label.append("face_"+str(xyz) + "_y")
                col_label.append("face_"+str(xyz) + "_z")
                holmesh_csv.append(face_landmark.x)
                holmesh_csv.append(face_landmark.y)
                holmesh_csv.append(face_landmark.z)
        if not results.face_landmarks:
            flag=0
            for xyz in range(468):
                col_label.append("face_"+str(xyz) + "_x")
                col_label.append("face_"+str(xyz) + "_y")
                col_label.append("face_"+str(xyz) + "_z")
                for _ in range(3):
                    #holmesh_csv.append(np.nan)
                    holmesh_csv.append("NAN")
        if results.left_hand_landmarks:
            flag=1
            for xyz, left_hand_landmark in enumerate(results.left_hand_landmarks.landmark):
                col_label.append("left_hand_"+str(xyz) + "_x")
                col_label.append("left_hand_"+str(xyz) + "_y")
                col_label.append("left_hand_"+str(xyz) + "_z")
                holmesh_csv.append(left_hand_landmark.x)
                holmesh_csv.append(left_hand_landmark.y)
                holmesh_csv.append(left_hand_landmark.z)
        if not results.left_hand_landmarks:
            flag=0
            for xyz in range(21):
                col_label.append("left_hand_"+str(xyz) + "_x")
                col_label.append("left_hand_"+str(xyz) + "_y")
                col_label.append("left_hand_"+str(xyz) + "_z")
                for _ in range(3):
                    #holmesh_csv.append(np.nan)
                    holmesh_csv.append("NAN")
        if results.right_hand_landmarks:
            flag=1
            for xyz, right_hand_landmark in enumerate(results.right_hand_landmarks.landmark):
                col_label.append("right_hand_"+str(xyz) + "_x")
                col_label.append("right_hand_"+str(xyz) + "_y")
                col_label.append("right_hand_"+str(xyz) + "_z")
                holmesh_csv.append(right_hand_landmark.x)
                holmesh_csv.append(right_hand_landmark.y)
                holmesh_csv.append(right_hand_landmark.z)
        if not results.right_hand_landmarks:
            flag=0
            for xyz in range(21):
                col_label.append("right_hand_"+str(xyz) + "_x")
                col_label.append("right_hand_"+str(xyz) + "_y")
                col_label.append("right_hand_"+str(xyz) + "_z")
                for _ in range(3):
                    #holmesh_csv.append(np.nan)
                    holmesh_csv.append("NAN")
        if results.pose_landmarks:
            flag=1
            for xyz, pose_landmark in enumerate(results.pose_landmarks.landmark):
                col_label.append("pose_"+str(xyz) + "_x")
                col_label.append("pose_"+str(xyz) + "_y")
                col_label.append("pose_"+str(xyz) + "_z")
                holmesh_csv.append(pose_landmark.x)
                holmesh_csv.append(pose_landmark.y)
                holmesh_csv.append(pose_landmark.z)
        if not results.pose_landmarks:
            flag=0
            for xyz in range(33):
                col_label.append("pose_"+str(xyz) + "_x")
                col_label.append("pose_"+str(xyz) + "_y")
                col_label.append("pose_"+str(xyz) + "_z")
                for _ in range(3):
                    #holmesh_csv.append(np.nan)
                    holmesh_csv.append("NAN")
      
      # 1枚目の画像をDataFrame構造で保存
      if idx == 0:
        data = pd.DataFrame([holmesh_csv], columns=col_label)
      # 1枚目と2枚目以降のDataFrameを縦に結合
      else:
        data1 = pd.DataFrame([holmesh_csv], columns=col_label)
        data = pd.concat([data, data1], ignore_index=True)
      
      try:
        if(flag==1):
          cv2.imwrite(dir_output_image + str(idx) + '.jpg', annotated_image)
      except UnboundLocalError:
        pass
      time.sleep(1)
  data.to_csv(dir_output_csv)

  return 

for i in range(len_sign):
    file_path="sign_"+str(i+1)
    for class_i in tqdm(list_class):
        #count=1
        basefile_inputpath=os.path.join(dir_input,class_i,file_path)
        basefile_outputpath=os.path.join(dir_output_image,class_i,file_path)
        dir_output_csvpath=os.path.join(dir_output_csv,class_i,file_path)
        #ディレクトリの中身分ループ
        for index in tqdm(range(len(os.listdir(basefile_inputpath)))):
          #print(file_name)
          hol_mediapipe_static(basefile_inputpath+r"\video_number"+str(index+1)+r"\*",
          basefile_outputpath+r"\video_number"+str(index+1),
          dir_output_csvpath+r"\videonum"+str(index+1)+r".csv")
          #count+=1
#hol_mediapipe_static(dir_input,dir_output_image,dir_output_csv)