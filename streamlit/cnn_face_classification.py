import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps , ImageDraw, ImageFont
import cv2
import io
import os
import json


st.set_option('deprecation.showfileUploaderEncoding', False)
filepathdesu='/Users/shinohararikunin/Desktop/model'
modeldesu = tf.keras.models.load_model(filepathdesu+'/Emotions_model.hdf5')

#modeldesu.load_weights(desu+'/mask.h5')

classes=["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

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

def pre_image(image_pil_array:'PIL.Image'):
	image_pil_array=get_square_image(image_pil_array)
	img=image_pil_array.resize((150,150))
	img=np.expand_dims(img,0)
	img=img/255.0
	#img_array=np.expand_dims(np.array(img).flatten()/255,0)
	img_array=np.array(img)
	result=modeldesu.predict(img_array)
	return 1-np.float(result[0][0])

def predict_emotions(image):
	img=image.reshape(1,48,48,3)
	img=img/255
	pred=modeldesu.predict(img)[0]
	top=3
	top_indices=pred.argsort()[-top:][::-1]
	result=[(classes[i],pred[i]) for i in top_indices]
	return result[0],result[1],result[2]




def get_result(prediction):
	'''0-1の数値を受け取って表示用のテキストを返す'''
	if prediction < 0.05:
		result = "確実にマスク"
	elif prediction < 0.2:
		result = "ほぼマスク"
	elif prediction < 0.5:
		result = "どちらかといえばマスク"
	elif prediction < 0.8:
		result = "どちらかといえばマスクなし"
	elif prediction < 0.95:
		result = "ほぼマスクなし"
	else:
		result = "確実にマスクなし"
	return result

def make_face_image(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    image = np.asarray(img)

    # 取得した長方形の座標を使って画像データを切り抜き
    face_image = image[top:bottom, left:right]

    # np.resizeだと画像が潰れちゃうのでcv2で読み取る
    cv2_img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # cv2はカラー画像をBGRの順番で読み取ってしまうのでアンパックで色要素を分けて代入
    b,g,r = cv2.split(cv2_img)

    # RGBの順番になるようにマージ（正しいカラー画像）
    face_image_color = cv2.merge([r,g,b])

    # モデルの学習を(128,128,3)でやってきたので縦横のサイズを128に揃える
    resized_image = cv2.resize(face_image_color, (48, 48))
    return resized_image

def main():
    st.title('表情分類器')

    st.write("cnn modelを使って、アップロードした画像を分類します。")

    uploaded_file = st.file_uploader('Choose a image file to predict')
	
    if uploaded_file is not None:
        image_pil_array = Image.open(uploaded_file)
        st.image(
            image_pil_array, caption='uploaded image',
            use_column_width=True
		)
		face_img=make_face_image(image_pil_array)
		first,second,third=predict_emotions(face_img)
		st.write('機械学習モデルは画像を', first, 'と予測しました。')
if __name__ == '__main__':
    main()