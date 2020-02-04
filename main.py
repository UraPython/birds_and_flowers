# libraries
import numpy as np
from shutil import copyfile
import os
import glob
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

# classify image with model
def predict_image(filename, model):
    original = load_img(filename,target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg16.preprocess_input(image_batch.copy())
    return decode_predictions(
        vgg_model.predict(processed_image))[0][0][1]
    
# load vgg model
vgg_model = vgg16.VGG16(weights='imagenet')

# flower classes
flowers = ['daisy', 'pot']

# bird classes
birds = [
    'goldfinch', 'European_gallinule', 'brambling',
    'peacock', 'indigo_bunting', 'lorikeet',
    'bulbul', 'great_grey_owl', 'hen', 
]

# folder name
FOLDER_NAME = "vacationImages"

# number of files
num_files = len(glob.glob(FOLDER_NAME + "/*.jpg"))

# create new folders
os.mkdir(FOLDER_NAME + "/flowers")
os.mkdir(FOLDER_NAME + "/birds")
os.mkdir(FOLDER_NAME + "/none")

for i in range(num_files):
    filename = FOLDER_NAME + "/" + str(i) + ".jpg"
    pred = predict_image(filename, vgg_model)
    if  pred in flowers:
        copyfile(filename, FOLDER_NAME + '/flowers/' + str(i) + ".jpg")
    elif pred in birds:
        copyfile(filename, FOLDER_NAME + '/birds/' + str(i) + ".jpg")
    else:
        copyfile(filename, FOLDER_NAME + '/none/' + str(i) + ".jpg")