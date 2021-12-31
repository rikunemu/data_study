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
	#image_pil_array=get_square_image(image_pil_array)
	img=image_pil_array.resize((48,48))
	img=np.expand_dims(img,0)
	#img=img/255.0
	#img_array=np.expand_dims(np.array(img).flatten()/255,0)
	img_array=np.array(img)
	input_shape=(48,48,1)
	#x_train = img_array.reshape(1, 48, 48)

	pred=modeldesu.predict(img)
	#top_indices = pred.argsort()[-3:][::-1]
	#result = [(faceclasses[i], pred[i]) for i in top_indices]

	#return result[0], result[1], result[2]
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
        st.write('機械学習モデルはAngryを', result[0][0], 'と予測しました。')
        st.write('機械学習モデルはDisgustを', result[0][1], 'と予測しました。')
        st.write('機械学習モデルはFearを', result[0][2], 'と予測しました。')
        st.write('機械学習モデルはHappyを', result[0][3], 'と予測しました。')
        st.write('機械学習モデルはNeutralを', result[0][4], 'と予測しました。')
        st.write('機械学習モデルはSadを', result[0][5], 'と予測しました。')
        st.write('機械学習モデルはSurpriseを', result[0][6], 'と予測しました。')

        


if __name__ == '__main__':
    main()