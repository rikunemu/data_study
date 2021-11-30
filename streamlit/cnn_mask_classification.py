import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps



st.set_option('deprecation.showfileUploaderEncoding', False)
desu='/Users/shinohararikunin/Desktop/model'
modeldesu = tf.keras.models.load_model(desu+'/model.h5')

modeldesu.load_weights(desu+'/mask.h5')



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

        
        
        pred =pre_image(image_pil_array)
        

        st.write('機械学習モデルは画像を', get_result(pred), 'と予測しました。')

        


if __name__ == '__main__':
    main()