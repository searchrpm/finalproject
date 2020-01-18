import os
import numpy as np
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from keras.applications.vgg19 import (
    VGG19, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import merge, Input
from keras.models import model_from_json
from keras.models import Model
from keras.utils import to_categorical, np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import set_session
from keras.models import load_model

import tensorflow as tf
#global sess
#global graph

sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

image_input = Input(shape=(224, 224, 3))
vgg = VGG19(input_tensor=image_input, include_top=False, weights=None)
model_ = Sequential()
for layer in vgg.layers:
    model_.add(layer)

# add new classification layer
model_.add(GlobalAveragePooling2D())
model_.add(Dense(10, activation='softmax'))
print(model_.summary())

model_.load_weights("model_weights.h5")


# vgg = VGG19(input_tensor=image_input, include_top=True, weights='imagenet')
# model2 = Sequential()
# for layer in vgg.layers[:-1]:
#     model2.add(layer)

# # freeze the feature extractors
# for layer in model2.layers:
#     layer.trainable = False

# # # add new classification layer
# # model.add(GlobalAveragePooling2D())
# model2.add(Dense(10, activation='softmax'))
# model2.load_weights('model_weights.h5')

def prepare_image(img):
    """
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    # Scale from 0 to 255
    img /= 255
    # Invert the pixels
    img = 1 - img
    # Flatten the image to an array of pixels
    img = img.flatten().reshape(-1, 28 * 28)
    """
    plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Return the processed feature array
    return x


def predict_on_img(img, model):
  
  #image_size = (224, 224)
  #img = cv2.resize(img, image_size, cv2.INTER_AREA)
  #x = np.expand_dims(x, axis=0)
  #x = preprocess_input(x)

  mapping = {'angkor wat': 0,
  'chichen itza': 1,
  'machu pichu': 2,
  'moai statues': 3,
  'petra': 4,
  'pyramids of giza': 5,
  'taj mahal': 6,
  'temple mount': 7,
  'the colosseum': 8,
  'the great wall of china': 9}

  class_map = {v: k for k, v in mapping.items()}
  with graph.as_default():
      set_session(sess)
      y = model.predict_classes(img)
  #return '_'.join(map(str, y))
  return class_map[y[0]]


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(request)

        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the file to the uploads folder
            file.save(filepath)

            # Load the saved image using Keras and resize it to the
            # mnist format of 28x28 pixels
            image_size = (224, 224)
            img = image.load_img(filepath, target_size=image_size) #, grayscale=True)

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Convert the 2D image to an array of pixel values
            #image_array = prepare_image(im)
            #print(image_array)

            return predict_on_img(x, model_)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run(debug=True)
