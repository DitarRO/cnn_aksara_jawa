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

UPLOAD_FOLDER = './upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = keras.models.load_model('base_model.h5')

# def predict_label(img_stream: bytes):
#     img = Image.open(img_stream).resize((224, 224)).convert('L')
#     # img.show()
#     # print(img)
# 	# i = image.load_img(img_path, target_size=(224,224))
#     i = image.img_to_array(img)/255.0
#     # print(i.shape)
#     i = i.reshape(224,224, 1)
#     p = model.predict(i)
#     classes_x=np.argmax(p,axis=1)
#     return dic[classes_x[0]]
# # def load():


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    data_gambar = None
    if request.method == 'POST':
        f = request.files['file'] 
        # f.save(f.filename)
        # print(f.filename)
        stream = f.stream
        data_gambar = base64.b64encode(stream.read()).decode()
    return render_template('index.html', data=data_gambar, hasil='ha')

if __name__ == '__main__':
    app.run()

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