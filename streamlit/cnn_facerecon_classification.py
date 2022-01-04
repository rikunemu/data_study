import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding', False)
desu='/Users/shinohararikunin/Desktop/model'
modeldesu = tf.keras.models.load_model(desu+'/facemodel.hdf5')

#modeldesu.load_weights(desu+'/faceweightepoch10.h5')

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


def pre_image(image_pil_array:'PIL.Image'):
	image_pil_array=get_square_image(image_pil_array)
	img=image_pil_array.resize((48,48))
	img=np.expand_dims(img,0)
	img=img/255.0
	pred=modeldesu.predict(img)
	return pred

def main():
    

    st.title('画像分類器')

    st.write("cnn modelを使って、アップロードした画像を分類します。")

    uploaded_file = st.file_uploader('Choose a image file to predict')

    if uploaded_file is not None:
        image_pil_array = Image.open(uploaded_file)
        st.image(
            image_pil_array, caption='uploaded image',
            use_column_width=True
        )
 
        result=pre_image(image_pil_array)
        #st.write('機械学習モデルは画像を', first, 'と予測しました。')
        st.slider('Angry:', 0.0, 1.0, float(result[0][0]), 0.01)
        st.slider('Disgust:', 0.0, 1.0, float(result[0][1]), 0.01)
        st.slider('Fear:', 0.0, 1.0, float(result[0][2]), 0.01)
        st.slider('Happy:', 0.0, 1.0, float(result[0][3]), 0.01)
        st.slider('Neutral:', 0.0, 1.0, float(result[0][4]), 0.01)
        st.slider('Sad:', 0.0, 1.0, float(result[0][5]), 0.01)
        st.slider('Surprise:', 0.0, 1.0, float(result[0][6]), 0.01)
        

if __name__ == '__main__':
    main()