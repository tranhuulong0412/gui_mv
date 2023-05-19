import time
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input


st.set_page_config(layout="wide")

st.title("Nhận diện bệnh Viêm phổi, Covid sử dụng CNN")

model = tf.keras.models.load_model('mv2.h5')

uploaded_file = st.file_uploader("Choose a X-ray image file", type=["jpg","jpeg","png"])


def histogram_equalization(image):
    # Chuyển đổi ảnh sang ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tính toán histogram của ảnh xám
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    # Tính toán histogram tích lũy
    cum_hist = hist.cumsum()
    
    # Chuẩn hóa histogram tích lũy
    cum_hist_normalized = cum_hist / cum_hist.max()
    
    # Ánh xạ lại các giá trị cường độ từ histogram tích lũy chuẩn hóa
    mapping = (cum_hist_normalized * 255).astype(np.uint8)
    
    # Áp dụng phép ánh xạ lại lên ảnh gốc
    equalized_image = mapping[gray_image.astype(np.uint8)]
    
    return equalized_image
    
if uploaded_file is not None:
    img = image.load_img(uploaded_file,target_size=(300,300))
    
    col1, col2 = st.columns(2) 
    with col1:
        st.write('**X-RAY IMAGE NON-PROCESS**')
        st.image(img, channels="RGB")
        Process = st.button("**Pre-process & Predict**")

    if Process:
        img_array = img_to_array(img)
        img_array = histogram_equalization(img_array)
        img_array = np.expand_dims(img_array, axis=-1)  # Thêm kích thước kênh màu cuối cùng
        img_array = np.repeat(img_array, 3, axis=-1)  # Lặp lại giá trị kênh màu để có kích thước (150, 150, 3)
        img = image.array_to_img(img_array)

        with col2:
            st.write('**X-RAY IMAGE IS PROCESSED**')
            st.image(img, channels="RGB")
            img = img.resize((150,150))
            img = img_to_array(img)
            img = img.reshape(1,150,150,3)
            img = img.astype('float32')
            img = img / 255

            with st.spinner("Waiting !!!"):
                time.sleep(2)

            result = int(np.argmax(model.predict(img),axis =1))
            percent = model.predict(img)

            if result == 0:
                st.write("**Based on the x-ray image it is COVID19**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ",percent,"%")
            elif result == 1 :
                st.write("**Based on the x-ray image it is HEALTHY**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ", percent,"%")
            else :
                st.write("**Based on the x-ray image it is PNEUMOIA**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ", percent,"%")

