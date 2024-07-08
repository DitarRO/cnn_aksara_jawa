import PIL.Image
from flask import Flask, render_template, request
import os
import base64
import tensorflow as tf
import keras
from keras.preprocessing import image
from PIL import Image
import io
import numpy as np
from keras.utils import img_to_array
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Input,Add,MaxPool2D,Flatten,AveragePooling2D,Dense,BatchNormalization,ZeroPadding2D,Activation,Concatenate,UpSampling2D
from tensorflow.keras.models import Model


classes = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma',
           'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

UPLOAD_FOLDER = './upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = keras.models.load_model('./model_91.keras')

def predict(img_buf):
    if img_buf:
        img_pil = Image.open(io.BytesIO(img_buf)).convert('RGB').resize(size=(224, 224))
    # img_krs = load_img('./contoh.png', color_mode='grayscale', target_size=(224, 224))
    # img_krs.show()
        img_arr = img_to_array(img_pil)
        x = np.expand_dims(img_arr, axis=0)

        test_img = np.vstack([x])
    # print(test_img)
        result = model.predict(test_img, batch_size=8)
    # print(image_path)
        return classes[np.argmax(result)]
    return "None"



@app.route('/', methods=['POST', 'GET'])
def upload_file():
    img_buf = None
    data_gambar = None
    if request.method == 'POST':
        f = request.files['file'] 
        stream = f.stream
        img_buf = stream.read()
        data_gambar = base64.b64encode(img_buf).decode()
    return render_template('index.html', data=data_gambar, hasil=predict(img_buf))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run()

# buffer = open('D:\!!WorkSpace\cnn_jawa\static\img\huruf_jawa.png', 'rb')
# buffer.write(open('D:\!!WorkSpace\cnn_jawa\static\img\contoh.png', 'rb').read())
# buffer.seek(0)
# print(len(buffer))
# model.summary()
# import numpy as np

# Muat model
# model = keras.models.load_model('base_model.h5')

# Buat data input dummy dengan shape yang sesuai
# input_shape = (224, 224, 1)
# dummy_input = np.zeros((1,) + input_shape)

# Panggil model dengan input dummy untuk mendefinisikan input
# model(dummy_input)

# Tampilkan ringkasan model setelah dipanggil
# model.layers

# predict_label(buffer)
