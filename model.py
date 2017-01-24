# Import the libraries we'll need
from keras.layers import ELU
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
import matplotlib.colors as c
import cv2
import csv
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

tf.python.control_flow_ops = tf

IMAGES_DIRECTORY = "./data/data"
DRIVING_LOG = "./data/data/driving_log.csv"


def load_driving_log(path):
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        rows = []
        for row in reader:
            row['center'] = row['center'].strip()
            row['left'] = row['left'].strip()
            row['right'] = row['right'].strip()
            row['throttle'] = float(row['throttle'])
            row['steering'] = float(row['steering'])
            row['brake'] = float(row['brake'])
            row['speed'] = float(row['speed'])

            rows.append(row)

    return rows


def resize(img, new_width=200, new_height=66):
    """
    Resizes an image. The return image will have size (new_width, new_height)
    """

    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def apply_random_brightness(img):
    img_hsv = c.rgb_to_hsv(img)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * (0.25 + np.random.uniform())
    return c.hsv_to_rgb(img_hsv)


def normalize(img):
    return img.astype("float") / 255


def crop_car_and_sky(img):
    return img[45:135, :, :]


def horizontal_flip(img):
    return cv2.flip(img, 1)


def pre_process(img, new_shape=(200, 66)):
    return normalize(resize(crop_car_and_sky(img), new_shape[0], new_shape[1]))


def augment_single_row(row):
    camera_selection = np.random.choice(['center', 'left', 'right'])
    img_path = IMAGES_DIRECTORY + "/" + row[camera_selection]
    angle = row['steering']

    if camera_selection == 'left':
        angle += 0.275
    elif camera_selection == 'right':
        angle -= 0.275

    image = mpimg.imread(img_path)
    # image = cv2.imread(img_path)
    image = pre_process(apply_random_brightness(image))
    # image = pre_process(image)

    flip = np.random.choice(['yes', 'no'])

    if flip == 'yes':
        image = horizontal_flip(image)
        angle *= -1

    return image, angle


def augment_batch(batch):
    X_batch, y_batch = [], []

    for row in batch:
        img, angle = augment_single_row(row)
        X_batch.append(img)
        y_batch.append(angle)

    return np.array(X_batch), np.array(y_batch)


def data_generator(rows, batch_size=16):
    """
    Creates a generator for the features and labels passed as input.
    """

    n_rows = len(rows)
    batches_per_epoch = n_rows // batch_size

    current_batch = 0

    while True:
        start = current_batch * batch_size
        end = start + batch_size

        batch = rows[start:end]

        X_batch, y_batch = augment_batch(batch)

        current_batch += 1

        if current_batch == batches_per_epoch:
            current_batch = 0

        yield X_batch, y_batch


def get_model(image_shape):
    """
    Build a neural network model for predicting steering angle given center camera images.
    """

    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init='normal', border_mode='valid', input_shape=image_shape))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init='normal', border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), init='normal', border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='normal', border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='normal', border_mode='valid'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(1164, init='normal'))
    model.add(ELU())

    model.add(Dense(100, init='normal'))
    model.add(ELU())

    model.add(Dense(50, init='normal'))
    model.add(ELU())

    model.add(Dense(1, init='normal'))

    return model


def plot_history(h):
    """
    Uses the history returned by the training process of a model to plot the evolution of the loss and accuracy.
    """

    # Let's print the metrics used during training
    print(h.history.keys())

    # Summarize history for loss
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def save_model(m):
    """
    Saves the model and its weights to 'model.json' and 'model.h5' files respectively.
    """

    # Save model & weights
    model_json = m.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    m.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    BATCH_SIZE = 64
    data = load_driving_log(DRIVING_LOG)
    TRAIN_SPLIT = 0.8
    split_point = int(len(data) * TRAIN_SPLIT)

    train_data = data[:split_point]
    validation_data = data[split_point:]

    model = get_model(image_shape=(66, 200, 3))
    train_generator = data_generator(data, batch_size=BATCH_SIZE)
    validation_generator = data_generator(validation_data, batch_size=BATCH_SIZE)

    # Compile our model
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Train our model
    history = model.fit_generator(train_generator, validation_data=validation_generator, nb_epoch=4,
                                  samples_per_epoch=50000, nb_val_samples=10000)

    save_model(model)
    plot_history(history)

