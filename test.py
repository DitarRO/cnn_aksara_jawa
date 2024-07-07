from PIL import Image
import io
import tensorflow as tf
from keras.api.utils import img_to_array
from keras.api.models import load_model
import numpy as np
from keras.api.preprocessing.image import load_img

classes = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma',
           'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

model = load_model('./base_model_anyar.h5')

def predict(img_buf):
    img_pil = Image.open(io.BytesIO(img_buf)).convert('L').resize(size=(224, 224))
    # img_krs = load_img('./contoh.png', color_mode='grayscale', target_size=(224, 224))
    # img_krs.show()
    img_arr = img_to_array(img_pil)
    x = np.expand_dims(img_arr, axis=0)

    test_img = np.vstack([x])
    # print(test_img)
    result = model.predict(test_img, batch_size=8)
    # print(image_path)
    return classes[np.argmax(result)]


if __name__ == "__main__":
    img_buf = open("./ba17.png", "rb")

    predict(img_buf.read())
