# Behavioral Cloning

Code base for a deep learning model purposed to teach a car to drive by itself, just by mimicking/cloning the behavior 
of a human at the wheel. This project was developed as part of the amazing [Self Driving Car Engineer Nanodegree 
offered by Udacity](https://www.udacity.com/drive)

## Data exploration

You can read more about the data exploration and familiarization process [here](https://github.com/jesus-a-martinez-v/behavioral-cloning/blob/master/exploration.ipynb).

### Recovery

One of the main problems with our data set is that it is biased towards driving straight (i.e. not turning at all). This
means that when the car approaches the limit of the road it won't steer because the aforementioned bias. So, what do we do?
Well, there are two techniques that can be used to prevent this situation:

* Record more images in the simulator **only** when driving from one of the borders of the road back to the center, or
* Use the images from the left and right cameras.

We used the second approach for the following reasons:

* Recording recovery data is **hard** and consumes **a lot** of time.
* In a real world setting is very unlikely that we'll be able to zigzag from one end to the road to the other just to teach our car how to recover (mainly because it is unsafe).

By using the left and right cameras images we can **simulate** that the car is driving back to the center just by adding a little steering angle (a positive one if we are coming from the left and a negative one if we're coming from the right). In our case, a steering angle of **+/- 0.275** gave the best results.

### Augmentation

Our data is insufficient, so we need to augment it. In order to do so we'll make use of two simple techniques:

 * **Horizontal flipping**: We flip the image around the Y axis and change the sign of the corresponding steering angle.
 * **Random brightness perturbation**: We add a little brightness perturbation to the images in order to emulate different lightning conditions. This is done mainly to add robustness to our model against shadows on the road.

### Pre-processing

We used [this GREAT model developed by NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (see next section). In order to stick to it we reshaped our input images 
 (with shape 160x320x3) to 66x200x3. The complete pre-processing pipeline consists of the following steps:
 
 * Crop the hood and most of the sky and scenery out of the image.
 * Reshape to 66x200x3.
 * Normalize pixels values, dividing them by 255.
 
Original image:

![alt tag](https://github.com/jesus-a-martinez-v/behavioral-cloning/blob/master/readme_assets/left.png)

After pre-processing:

![alt tag](https://github.com/jesus-a-martinez-v/behavioral-cloning/blob/master/readme_assets/preprocess.png)

## Model architecture

One of the **hardest** parts of solving a problem using machine learning and deep learning in particular is to determine
the correct architecture. Fortunately there are incredibly smart people working in all sort of problems. Even better, they share their findings and techniques so more and more people can use their knowledge and ideas as a starting point. That's exactly what we did. We used the model that the amazing NVIDIA team developed and published in this [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). Here's an outline of the architecture implemented:

![alt tag](https://github.com/jesus-a-martinez-v/behavioral-cloning/blob/master/readme_assets/1-oYI-6Ne_RfQcBiNftqcvew.png)

* **First layer**: Convolutional layer with 24x5x5 filters. Dropout of 0.5. ELU activation. 2x2 subsampling.
* **Second layer**: Convolutional layer with 36x5x5 filters. Dropout of 0.5. ELU activation. 2x2 subsampling.
* **Third layer**: Convolutional layer with 48x5x5 filters. Dropout of 0.5. ELU activation. 2x2 subsampling.
* **Fourth layer**: Convolutional layer with 64x3x3 filters. Dropout of 0.5. ELU activation. 1x1 subsampling.
* **Fifth layer**: Convolutional layer with 64x3x3 filters. Dropout of 0.5. ELU activation. 1x1 subsampling.
+ **Sixth layer**: Flatten layer.
* **Seventh layer**: Fully connected layer. 1164 units. ELU activation.
* **Eighth layer**: Fully connected layer. 100 units. ELU activation.
* **Ninth layer**: Fully connected layer. 50 units. ELU activation.
* **Tenth layer**: Fully connected layer. 10 units. ELU activation.
* **Eleventh layer**: Output layer. 1 unit.

We used ELU activation instead of ReLUs because, according to this article, it behaves better with negative and near-zero values. Hence, it speeds up the learning process a bit by preventing dead neurons when some of the weights gets equals to zero.

We used dropout between convolutional layers because here's where most of the connections are, so by dropping half of our activations each time we "force" our model to generalize better.

## Training

To train our model we used an Adam Optimizer because it is a good default. Also, this kind of optimizer is better than Gradient Descent because it usually converges faster by keeping track of the momentum. It adjusts the learning rate so our model doesn't overshoot trying to find the right direction downwards.

We split the data in two: 80% of the data were used to train the model and the remaining 20% for validation after each epoch.

In both cases (i.e., for training and validation) we used a Python generator that creates a batch of augmented data on demand. The batch sized used was 64. The augmentation process followed by the generator is:

* Select an image from **one** of the cameras (left, center or right).
* If the image is from the right camera, then subtract 0.275 from the steering angle.
* If the image is from the left camera, then add 0.275 to the steering angle.
* Pre-process the image.
* Apply random brightness perturbance to the image.
* "Flip a coin". If it's heads ("yes"), then flip the image around the Y axis and change the sign of the steering angle. If it is tails ("no") don't modify the image.

The model was trained using the Keras' `fit_generator` method, during 5 epochs.

50.000 images are generated for training and 10.000 for validation during each epoch.

## Results

You can see how the car performs on a relatively simple circuit [here](https://drive.google.com/open?id=0B1SO9hJRt-hgUm9TZzhjeDRyNmM).

[Here](https://drive.google.com/open?id=0B1SO9hJRt-hgOVl4TnprN08tMGs) you can watch the car perform on an unseen track, which presents more challenges, such as:

* More shadows.
* More curves to the right.
* More pronounced curves.
* Irregular terrain.
