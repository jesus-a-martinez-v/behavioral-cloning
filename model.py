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

# Fix error with Keras and TensorFlow
tf.python.control_flow_ops = tf

# Handy constants. Change accordingly
IMAGES_DIRECTORY = "./data/data"
DRIVING_LOG = "./data/data/driving_log.csv"


def load_driving_log(path):
    """
    Loads the driving log and stores its data in a list of dicts, where each dict represents a row.
    :param path: Path of the driving log file to load.
    :return: List of dicts.
    """
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        rows = []

        # Apply some minor processing to each field.
        for row in reader:
            # Must strip out extra blank characters from the path of the images.
            row['center'] = row['center'].strip()
            row['left'] = row['left'].strip()
            row['right'] = row['right'].strip()

            # Parse numeric fields into float numbers.
            row['throttle'] = float(row['throttle'])
            row['steering'] = float(row['steering'])
            row['brake'] = float(row['brake'])
            row['speed'] = float(row['speed'])

            rows.append(row)

    return rows


def resize(img, new_shape=(200, 66)):
    """
    Resizes an image. The return image will have size (new_shape[0], new_shape[1])
    """

    return cv2.resize(img, new_shape, interpolation=cv2.INTER_CUBIC)


def apply_random_brightness(img):
    """
    Applies random brightness modification to the input image.
    :param img: Image to be modified.
    :return: Input image with a little brightness perturbation (this means it may be a bit darker or a bit brighter).
    """
    # We must convert to HSV color space in order to successfully apply a random brightness perturbation.
    img_hsv = c.rgb_to_hsv(img)

    # Only alter the H and S channels.
    # The 0.25 is a constant that prevents the image from going completely black.
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * (0.25 + np.random.uniform())

    # We must come back to RGB space.
    return c.hsv_to_rgb(img_hsv)


def normalize(img):
    """
    Takes an image and normalizes each of its pixel values by dividing their value by 255. Hence, each value will range
    between 0 and 1.
    :param img: Image to be normalized.
    :return: Input image normalized so its pixel values belong to the interval [0, 1]
    """
    return img.astype("float") / 255


def crop_car_and_sky(img):
    """
    Takes out the portion of the image where the hood is visible, as well as a fair amount of the sky and other
    landscape elements.
    :param img: Image to be modified.
    :return: Input image without the segments where (most of) the sky and car's hood were visible.
    """
    return img[45:135, :, :]


def horizontal_flip(img):
    """
    Takes an image and flips it around the Y axis.
    :param img: Image to be flipped.
    :return: Input image horizontally flipped.
    """
    return cv2.flip(img, 1)


def pre_process(img, new_shape=(200, 66)):
    """
    Takes an image and pre process it by applying the following transformations:
        1. Cropping.
        2. Resizing.
        3. Normalization.
    :param img: Image to be processed.
    :param new_shape: Shape of the output image.
    :return: Input imaged after the processing steps described above.
    """
    return normalize(resize(crop_car_and_sky(img), new_shape))


def augment_single_row(row):
    """
    Takes a single row of the driving log and modifies ONE OF the images from ONE OF the cameras.
    :param row: dict containing the data of a single row of the driving log file.
    :return: Augmented/modified image and its corresponding label (steering angle).
    """

    # We'll select an image from one of the three cameras.
    camera_selection = np.random.choice(['center', 'left', 'right'])
    img_path = IMAGES_DIRECTORY + "/" + row[camera_selection]

    angle = row['steering']

    # This step is meant to augment the image for recovery. So, if the image we selected isn't from the center camera,
    # we'll add a little constant to the steering angle so the car can head back to the center of the track when it
    # approaches too much to the edge of the road.
    if camera_selection == 'left':
        angle += 0.275  # Must steer right to get back to the center.
    elif camera_selection == 'right':
        angle -= 0.275  # Must steer left to get back to the center.

    # Let's load the actual image.
    image = mpimg.imread(img_path)
    # Pre-process the image after applying a random brightness perturbation.
    image = pre_process(apply_random_brightness(image))

    # Flip a coin in order to now if we should flip the image. This is done because our training track has more
    # left turns than right turns, so we need to balance our data.
    flip = np.random.choice(['yes', 'no'])

    if flip == 'yes':
        image = horizontal_flip(image)
        angle *= -1

    return image, angle


def augment_batch(batch):
    """
    Takes a batch (a cluster of rows from the driving log, actually) and augments it.
    :param batch: List of dicts to be augmented.
    :return: Images and labels for this batch.
    """
    X_batch, y_batch = [], []

    # Augment each row of the batch.
    for row in batch:
        img, angle = augment_single_row(row)
        X_batch.append(img)
        y_batch.append(angle)

    # Return as numpy arrays.
    return np.array(X_batch), np.array(y_batch)


def data_generator(rows, batch_size=16):
    """
    Creates a generator for the features and labels passed as input.
    """

    n_rows = len(rows)
    batches_per_epoch = n_rows // batch_size

    current_batch = 0

    # Generate data until the end of time...
    while True:
        # Find the beginning and the end of the batch.
        start = current_batch * batch_size
        end = start + batch_size

        # Get the actual batch.
        batch = rows[start:end]

        # Augment it
        X_batch, y_batch = augment_batch(batch)

        # Move the cursor to the next batch.
        current_batch += 1

        # If we exhausted all the data, we must start all over again.
        if current_batch == batches_per_epoch:
            current_batch = 0

        # Serve the batch.
        yield X_batch, y_batch


def get_model(image_shape):
    """
    Build a neural network model for predicting steering angle given center camera images. The actual architecture
    was borrowed from this AMAZING paper by Nvidia:
    - http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """

    # Since more of our connections are between the convolutional layers we use dropout only between convolutions.

    # We use ELU instead of ReLU because it behaves better with negative weights, so we avoid dead neurons.
    # More info here: http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/
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

    model.add(Dense(1, init='normal'))  # Just one neuron because we only need the steering angle.

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

    # Load the data.
    data = load_driving_log(DRIVING_LOG)

    # np.random.seed(42)
    np.random.shuffle(data)  # Shuffle the data so we can do a better training/validation split

    TRAIN_SPLIT = 0.8  # 80% of the data for training, 20% for validation.
    split_point = int(len(data) * TRAIN_SPLIT)

    # Split
    train_data = data[:split_point]
    validation_data = data[split_point:]

    # Get model. We're reshaping images to 200x66 so we conform to the NVidia model.
    model = get_model(image_shape=(66, 200, 3))

    # Get generators for each data set.
    train_generator = data_generator(data, batch_size=BATCH_SIZE)
    validation_generator = data_generator(validation_data, batch_size=BATCH_SIZE)

    # Compile our model
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Train our model
    history = model.fit_generator(train_generator, validation_data=validation_generator, nb_epoch=5,
                                  samples_per_epoch=50000, nb_val_samples=10000)

    save_model(model)
    plot_history(history)

