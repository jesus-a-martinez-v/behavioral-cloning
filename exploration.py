import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import numpy as np
import model
import cv2
from keras.preprocessing.image import img_to_array, load_img

IMAGES_DIRECTORY = "./data/data"
DRIVING_LOG = "./data/data/driving_log.csv"

rows = model.load_driving_log(DRIVING_LOG)

img = mpimg.imread(IMAGES_DIRECTORY + "/" + rows[0]['center'])
# img = img_to_array(load_img(IMAGES_DIRECTORY + "/" + rows[0]['center'])).astype(np.float32)
plt.imshow(model.pre_process(model.apply_random_brightness(model.horizontal_flip(img))))
plt.show()
