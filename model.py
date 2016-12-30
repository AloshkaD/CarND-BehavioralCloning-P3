import numpy as np
import pandas as pd
import json
import uuid
import os
import random
import cv2
import math

from keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy import misc
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import tensorflow as tf


class RecordingMeasurement:
    """
    A representation of a vehicle's state at a point in time while driving
    around a track during recording.

    Features available are:
        
        left_camera_view_path   - Path to the physical image taken by the LEFT camera.

        center_camera_view_path - Path to the physical image taken by the CENTER camera.

        right_camera_view_path  - Path to the physical image taken by the RIGHT camera.

        left_camera_view()      - An image taken by the LEFT camera.

        center_camera_view()    - An image taken by the CENTER camera.

        right_camera_view()     - An image taken by the RIGHT camera.

        steering_angle          - A normalized steering angle in the range -1 to 1.

        speed                   - The speed in which the vehicle was traveling at measurement time.

    This class serves the following purposes:

      * Provides convenience getter methods for left, center and camera images.
         In an effort to reduce memory footprint, they're essentially designed
         to lazily instantiate (once) the actual image array at the time the
         method is invoked.

      * Strips whitespace off the left, center, and right camera image paths.

      * Casts the original absolute path of each camera image to a relative path.
         This adds reassurance the image will load on any computer.

      * Provides a convenient #is_valid_measurment method which encapsulates
         pertinent logic to ensure data quality is satisfactory.

    """

    def __init__(self, measurement_data):
        self.measurement_data = measurement_data

        self.steering_angle = round(float(measurement_data['steering']), 4)
        self.speed = round(float(measurement_data['speed']), 4)

        l = measurement_data['left'].strip()
        c = measurement_data['center'].strip()
        r = measurement_data['right'].strip()

        # cast absolute path to relative path to be environment agnostic
        l, c, r = [(os.path.join(os.path.dirname(__file__), 'IMG', os.path.split(file_path)[1])) for file_path in
                   (l, c, r)]

        self.left_camera_view_path = l
        self.center_camera_view_path = c
        self.right_camera_view_path = r

    def is_valid_measurement(self):
        """
        Return true if the original center image is available to load.
        """
        return os.path.isfile(self.center_camera_view_path)

    def left_camera_view(self):
        """
        Lazily instantiates the left camera view image at first call.
        """
        if not hasattr(self, '__left_camera_view'):
            self.__left_camera_view = self.__load_image(self.left_camera_view_path)
        return self.__left_camera_view

    def center_camera_view(self):
        """
        Lazily instantiates the center camera view image at first call.
        """
        if not hasattr(self, '__center_camera_view'):
            self.__center_camera_view = self.__load_image(self.center_camera_view_path)
        return self.__center_camera_view

    def right_camera_view(self):
        """
        Lazily instantiates the right camera view image at first call.
        """
        if not hasattr(self, '__right_camera_view'):
            self.__right_camera_view = self.__load_image(self.right_camera_view_path)
        return self.__right_camera_view

    def __load_image(self, imagepath):
        image_array = None
        if os.path.isfile(imagepath):
            image_array = misc.imread(imagepath)
        else:
            print('File Not Found: {}'.format(imagepath))
        return image_array

    def __str__(self):
        results = []
        results.append('Image paths:')
        results.append('')
        results.append('     Left camera path: {}'.format(self.left_camera_view_path))
        results.append('   Center camera path: {}'.format(self.center_camera_view_path))
        results.append('    Right camera path: {}'.format(self.right_camera_view_path))
        results.append('')
        results.append('Additional features:')
        results.append('')
        results.append('   Steering angle: {}'.format(self.steering_angle))
        results.append('            Speed: {}'.format(self.speed))
        return '\n'.join(results)


def preprocess_image(image_array, output_shape=(20, 40), colorspace='yuv'):
    """
    Reminder:

    Source image shape is (160, 320, 3)

    My preprocessing algorithm consists of the following steps:

      1. Converts BGR to YUV colorspace.

         This allows us to leverage luminance (Y channel - brightness - black and white representation),
         and chrominance (U and V - blue–luminance and red–luminance differences respectively)

      2. Crops top 31.25% portion and bottom 12.5% portion.
         The entire width of the image is preserved.

         This allows the model to generalize better to unseen roadways since we crop
         artifacts such as trees, buildings, etc. above the horizon. We also clip the
         hood from the image.

      3. Finally, I allow users of this algorithm the ability to specify the shape of the final image via
         the output_shape argument.

         Once I've cropped the image, I resize it to the specified shape using the INTER_AREA
         interpolation algorithm as it is the best choice to preserve original image features.

         See `Scaling` section in OpenCV documentation:

         http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
    """
    # convert image to another colorspace
    if colorspace == 'yuv':
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2YUV)
    elif colorspace == 'hsv':
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    elif colorspace == 'hls':
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2HLS)
    elif colorspace == 'rgb':
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # [y1:y2, x1:x2]
    #
    # crops top 40% portion and bottom 12.5% portion
    #
    # The entire width of the image is preserved
    image_array = image_array[65:140, 0:320]

    # Let's blur the image to smooth out some of the artifacts
    kernel_size = 5  # Must be an odd number (3, 5, 7...)
    image_array = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)

    # resize image to output_shape
    image_array = cv2.resize(image_array, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA)

    return image_array


class Track1Dataset:
    """
    Parses driving_log.csv and constructs training, validation and test datasets corresponding to
    measurements taken at various points in time while recording on track 1.

        * X_train - A set of examples used for learning, that is to fit the parameters [i.e., weights] of the
                    classifier.

        * X_val - A set of examples used to tune the hyperparameters [i.e., architecture, not weights] of a
                       classifier, for example to choose the number of hidden units in a neural network.

        * X_test - A set of examples used only to assess the performance [generalization] of a fully-specified
                   classifier.

        * y_train, y_val, y_test - The steering angle corresponding to their respective X features.
    """

    DRIVING_LOG_PATH = './driving_log.csv'

    def __init__(self, validation_split_percentage=0.2, test_split_percentage=0.05):
        self.X_train = []
        self.X_val = []
        self.X_test = []

        self.y_train = []
        self.y_val = []
        self.y_test = []

        self.dataframe = None
        self.headers = []
        self.__loaded = False

        self.__load(validation_split_percentage=validation_split_percentage,
                    test_split_percentage=test_split_percentage)

        assert self.__loaded == True, 'The dataset was not loaded. Perhaps driving_log.csv is missing.'

    def __load(self, validation_split_percentage, test_split_percentage):
        """
        Splits the training data into a validation and test dataset.

        * X_train - A set of examples used for learning, that is to fit the parameters [i.e., weights] of the classifier.

        * X_val - A set of examples used to tune the hyperparameters [i.e., architecture, not weights] of a
                       classifier, for example to choose the number of hidden units in a neural network.

        * X_test - A set of examples used only to assess the performance [generalization] of a fully-specified
                   classifier.

        * y_train, y_val, y_test - The steering angle corresponding to their respective X features.
        """
        if not self.__loaded:
            X_train, y_train, headers, df = [], [], [], None

            # read in driving_log.csv and construct the
            # initial X_train and y_train before splitting
            # it into validation and testing sets.
            if os.path.isfile(self.DRIVING_LOG_PATH):
                df = pd.read_csv(self.DRIVING_LOG_PATH)
                headers = list(df.columns.values)
                for index, measurement_data in df.iterrows():
                    measurement = RecordingMeasurement(measurement_data=measurement_data)
                    if measurement.is_valid_measurement():
                        X_train.append(measurement)
                        y_train.append(measurement.steering_angle)
                    else:
                        print('FILE NOT FOUND')
                self.__loaded = True

            # generate the validation set
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=validation_split_percentage,
                random_state=0)

            X_train, y_train, X_val, y_val = np.array(X_train), np.array(y_train, dtype=np.float32), \
                                             np.array(X_val), np.array(y_val, dtype=np.float32)

            # generate the test set
            X_train, X_test, y_train, y_test = train_test_split(
                X_train,
                y_train,
                test_size=test_split_percentage,
                random_state=0)

            X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train, dtype=np.float32), \
                                               np.array(X_test), np.array(y_test, dtype=np.float32)

            self.X_train = X_train
            self.X_val = X_val
            self.X_test = X_test

            self.y_train = y_train
            self.y_val = y_val
            self.y_test = y_test

            self.dataframe = df
            self.headers = headers

    def batch_generator(self, X, Y, label, num_epochs, batch_size=32, output_shape=(160, 320),
                        classifier=None, colorspace='yuv'):
        """
        A custom batch generator with the main goal of reducing memory footprint
        on computers and GPUs with limited memory space.

        All left, center and right images are considered in the algorithm.

        Infinitely yields `batch_size` elements from the X and Y datasets.

        During batch iteration, this algorithm does the following:

          * Considers left, center and right images at each iteration based on the outcome of randomly generating a
            float from 0 to 1.

          * Finds the mean of all steering angles and augments 30% of all center camera steering angles, 40% of all
            left and right camera steering samples and 30% of center camera steering samples with no augmentation.

            Here is a more thorough breakdown of what the algorithm does for each sample:

            * left steering angles < 0.01 from the mean:

              * ~20% of all left curves get a 2x augmented steering angle with right camera image
              * ~20% of all left curves get a 1.5x augmented steering angle with right camera image
              * ~30% of all left curves get a 1.5x augmented steering angle with center camera image
              * ~30% of all left curves get actual steering angle with center camera image

            * right steering angles > 0.01 from the mean:

              * ~20% of all right curves get a 2x augmented steering angle with left camera image
              * ~20% of all right curves get a 1.5x augmented steering angle with left camera image
              * ~30% of all right curves get a 1.5x augmented steering angle with center camera image
              * ~30% of all right curves get actual steering angle with center camera image

            * All samples in {-0.01 < np.mean(steering_angles) < 0.01} are trained using the true steering angle and
              center camera image.

          * Passes the selected camera image to preprocess_image for further augmentation

          * randomly flips the image and steering angle 50% of the time to reduce bias towards a
            specific steering angle/direction
        """
        _population = len(X)
        _counter = 0
        _index_in_epoch = 0
        _batch_size = min(batch_size, _population)
        _batch_count = int(math.ceil(_population / _batch_size))

        assert X.shape[0] == Y.shape[0], 'X and Y size must be identical.'

        print('Batch generating against the {} dataset with population {} and shape {}'.format(label, _population,
                                                                                               X.shape))
        while True:
            _counter += 1
            print('batch gen iter {}'.format(_counter))
            for i in range(_batch_count):
                start_i = _index_in_epoch
                _index_in_epoch += _batch_size
                # all items have been seen; reshuffle and reset counters
                if _index_in_epoch >= _population:
                    # Save the classifier to support manual early stoppage
                    if classifier is not None:
                        classifier.save()
                    print('  sampled entire population. reshuffling deck and resetting all counters.')
                    perm = np.arange(_population)
                    np.random.shuffle(perm)
                    X = X[perm]
                    Y = Y[perm]
                    start_i = 0
                    _index_in_epoch = _batch_size
                end_i = min(_index_in_epoch, _population)

                X_batch = []
                y_batch = []

                y_mean = np.mean(Y)
                thresh = abs(y_mean) * 0.01
                l_thresh = y_mean - thresh
                r_thresh = y_mean + thresh

                for j in range(start_i, end_i):
                    steering_angle = Y[j]
                    measurement = X[j]

                    if steering_angle < l_thresh:
                        chance = random.random()

                        # 20% of the left curves get a 2x augmented steering angle with right camera image
                        if chance > 0.8:
                            image_array = measurement.right_camera_view()
                            augmented_steering = steering_angle * 2.0
                            steering_angle = augmented_steering
                        else:
                            # 20% of the left curves get a 1.5x augmented steering angle with right camera image
                            if chance > 0.6:
                                image_array = measurement.right_camera_view()
                                augmented_steering = steering_angle * 1.5
                                steering_angle = augmented_steering
                            else:
                                # 30% of left curves get a 1.5x augmented steering angle with center camera image
                                if chance < 0.3:
                                    image_array = measurement.center_camera_view()
                                    augmented_steering = steering_angle * 1.5
                                    steering_angle = augmented_steering

                                # 30% of left curves get actual steering angle with center camera image
                                else:
                                    image_array = measurement.center_camera_view()

                    if steering_angle > r_thresh:
                        chance = random.random()

                        # 20% of all right curves get a 2x augmented steering angle with left camera image
                        if chance > 0.8:
                            image_array = measurement.left_camera_view()
                            augmented_steering = steering_angle * 2.0
                            steering_angle = augmented_steering
                        else:
                            # 20% of all right curves get a 1.75x augmented steering angle with left camera image
                            if chance > 0.6:
                                image_array = measurement.left_camera_view()
                                augmented_steering = steering_angle * 1.75
                                steering_angle = augmented_steering
                            else:
                                # 30% of all right curves get a 1.5x augmented steering angle with center camera image
                                if chance < 0.3:
                                    image_array = measurement.center_camera_view()
                                    augmented_steering = steering_angle * 1.5
                                    steering_angle = augmented_steering

                                # 30% of all right curves get actual steering angle with center camera image
                                else:
                                    image_array = measurement.center_camera_view()
                    else:
                        image_array = measurement.center_camera_view()

                    image = preprocess_image(image_array, output_shape=output_shape, colorspace=colorspace)

                    # Here I throw in a random image flip to reduce bias towards
                    # a specific direction/steering angle.
                    if random.random() > 0.5:
                        X_batch.append(np.fliplr(image))
                        y_batch.append(-steering_angle)
                    else:
                        X_batch.append(image)
                        y_batch.append(steering_angle)

                yield np.array(X_batch), np.array(y_batch)

    def __str__(self):
        results = []
        results.append('{} Stats:'.format(self.__class__.__name__))
        results.append('')
        results.append('  [Headers]')
        results.append('')
        results.append('    {}'.format(self.headers))
        results.append('')
        results.append('')
        results.append('  [Shapes]')
        results.append('')
        results.append('    training features: {}'.format(self.X_train.shape))
        results.append('    training labels: {}'.format(self.y_train.shape))
        results.append('')
        results.append('    validation features: {}'.format(self.X_val.shape))
        results.append('    validation labels: {}'.format(self.y_val.shape))
        results.append('')
        results.append('    test features: {}'.format(self.X_test.shape))
        results.append('    test labels: {}'.format(self.y_test.shape))
        results.append('')
        results.append('  [Dataframe sample]')
        results.append('')
        results.append(str(self.dataframe.head(n=5)))
        return '\n'.join(results)


def load_dataset():
    dataset = Track1Dataset()
    print(dataset)
    return dataset


def visualize_dataset(dataset):
    dataset.dataframe.plot.hist(alpha=0.5)
    dataset.dataframe['steering'].plot.hist(alpha=0.5)
    dataset.dataframe['steering'].plot(alpha=0.5)


class BaseNetwork:
    def __init__(self):
        self.uuid = uuid.uuid4()
        self.model = None
        self.weights = None
        self.__model_file_name = 'model_{}.json'.format(self.__class__.__name__)
        self.__weights_file_name = self.__model_file_name.replace('json', 'h5')
        self.output_shape = None

    def build_model(self, input_shape=(160, 320, 3), learning_rate=0.001, dropout_prob=0.5, activation='elu'):
        raise NotImplementedError

    def fit(self, batch_generator, X_train, y_train, X_val, y_val,
            nb_epoch,
            batch_size,
            output_shape=(160, 320, 3),
            colorspace='yuv'):
        self.output_shape = output_shape

        # Keras throws an exception if we specify a batch generator
        # for an empty validation dataset.
        validation_data = None
        if len(X_val) > 0:
            validation_data = batch_generator(
                X=X_val,
                Y=y_val,
                label='validation set',
                num_epochs=nb_epoch,
                batch_size=batch_size,
                output_shape=output_shape,
                colorspace=colorspace
            )

        #checkpoint
        filepath = self.__class__.__name__+"-weights-improvement-{epoch:02d}-{val_loss:.4f}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        #early stopping
        earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=1, mode='min')

        callbacks_list = [checkpoint, earlystop]

        # Fit the model leveraging the custom
        # batch generator baked into the
        # dataset itself.
        history = self.model.fit_generator(
            batch_generator(
                X=X_train,
                Y=y_train,
                label='train set',
                num_epochs=nb_epoch,
                batch_size=batch_size,
                output_shape=output_shape,
                classifier=self,
                colorspace=colorspace
            ),
            nb_epoch=nb_epoch,
            samples_per_epoch=len(X_train),
            nb_val_samples=len(X_val),
            verbose=2,
            validation_data=validation_data,
            callbacks=callbacks_list
        )

        print(history.history)
        self.save()

    def save(self):
        self.__persist()
        print('Saved {} model.'.format(self.__class__.__name__))

    def restore(self):
        model = None
        if os.path.exists(self.__model_file_name):
            with open(self.__model_file_name, 'r') as jfile:
                the_json = json.load(jfile)
                print(json.loads(the_json))
                model = model_from_json(the_json)
            if os.path.exists(self.__weights_file_name):
                model.load_weights(self.__weights_file_name)
        return model

    def __persist(self):
        self.model.save_weights(self.__weights_file_name)
        with open(self.__model_file_name, 'w') as outfile:
            json.dump(self.model.to_json(), outfile)

    def __str__(self):
        results = []
        if self.model is not None:
            results.append(self.model.summary())
        return '\n'.join(results)


class Nvidia(BaseNetwork):
    NETWORK_NAME = 'nvidia'

    def fit(
            self,
            batch_generator, X_train, y_train, X_val, y_val,
            nb_epoch,
            batch_size,
            output_shape=(66, 200, 3),
            colorspace='yuv'
    ):
        super(Nvidia, self).fit(
            batch_generator, X_train, y_train, X_val, y_val,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            output_shape=output_shape,
            colorspace=colorspace
        )

    def build_model(self, input_shape=(66, 200, 3), learning_rate=0.001, dropout_prob=0.5, activation='elu',
                    use_weights=False):
        model = None
        if use_weights:
            model = self.restore()
        if model is None:
            model = Sequential()
            model.add(Lambda(lambda x: x / 255 - 0.5,
                             input_shape=input_shape,
                             output_shape=input_shape))
            model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation=activation))
            model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation=activation))
            model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation=activation))
            model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation=activation))
            model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation=activation))
            model.add(Flatten())
            model.add(Dropout(dropout_prob))
            model.add(Dense(1164, activation=activation))
            model.add(Dropout(dropout_prob))
            model.add(Dense(100, activation=activation))
            model.add(Dense(50, activation=activation))
            model.add(Dense(1, activation=activation))

        optimizer = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.model = model
        model.summary()
        return model


class Track1(BaseNetwork):
    """
        The Track1 network includes 5 convolution layers and an ELU activation that introduce non-linearity into
        the model.

        This network also includes a zero-mean normalization operator as the input layer. That will accelerate the
        convergence of the model to the solution.

        A 50% dropout operator that reduces the chance for overfitting by preventing units from co-adapting too much is
        used after flattening the convolution layers.

        A 50% dropout operator that reduces the chance for overfitting by preventing units from co-adapting too much is
        used after the first fully connected dense layer.

        Adam optimizer with 0.001 learning rate (default) used in this network.

        Mean squared error loss function was used since this is a regression problem and MSE is quite common and robust
        for regression analysis.
    """

    NETWORK_NAME = 'track1'

    def fit(
            self,
            batch_generator, X_train, y_train, X_val, y_val,
            nb_epoch=2,
            batch_size=32,
            output_shape=(20, 40, 3),
            colorspace='yuv'
    ):
        super(Track1, self).fit(
            batch_generator, X_train, y_train, X_val, y_val,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            output_shape=output_shape,
            colorspace=colorspace
        )

    def build_model(
            self,
            input_shape=(20, 40, 3),
            learning_rate=0.001,
            dropout_prob=0.5,
            activation='elu',
            use_weights=False
    ):
        """
        Initial zero-mean normalization input layer.
        A 4-layer deep neural network with 4 fully connected layers at the top.
        ELU activation used on each convolution layer.
        Dropout of 50% (default) used after initially flattening after convolution layers.
        Dropout of 50% (default) used after first fully connected layer.

        Adam optimizer with 0.001 learning rate (default) used in this network.
        Mean squared error loss function was used since this is a regression problem and MSE is
        quite common and robust for regression analysis.
        """
        model = None
        if use_weights:
            model = self.restore()
        if model is None:
            model = Sequential()
            model.add(Lambda(lambda x: x / 255 - 0.5,
                             input_shape=input_shape,
                             output_shape=input_shape))
            model.add(Dropout(dropout_prob))
            model.add(Convolution2D(24, 5, 5, border_mode='valid', activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Convolution2D(36, 5, 5, border_mode='valid', activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Convolution2D(48, 5, 5, border_mode='same', activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Convolution2D(64, 3, 3, border_mode='same', activation=activation))
            model.add(Flatten())
            model.add(Dropout(dropout_prob))
            model.add(Dense(1024, activation=activation))
            model.add(Dropout(dropout_prob))
            model.add(Dense(100, activation=activation))
            model.add(Dropout(dropout_prob))
            model.add(Dense(50, activation=activation))
            model.add(Dropout(dropout_prob))
            model.add(Dense(10, activation=activation))
            model.add(Dropout(dropout_prob))
            model.add(Dense(1, activation=activation, init='normal'))

        optimizer = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.model = model
        model.summary()
        return model


def train_network(
        classifier='track1',
        nb_epoch=2,
        batch_size=32,
        learning_rate=0.001,
        dropout_prob=0.5,
        activation='elu',
        use_weighs=False,
        colorspace='yuv'
):
    dataset = load_dataset()
    assert len(dataset.X_train) > 0, 'There is no training data available to train against.'
    if len(dataset.X_train) > 0:
        print('Center camera view shape:\n\n{}\n'.format(dataset.X_train[0].center_camera_view().shape))
        print(dataset.X_train[0])

    # instantiate proper classifier
    if classifier.lower() == Track1.NETWORK_NAME:
        clf = Track1()
    elif classifier.lower() == Nvidia.NETWORK_NAME:
        clf = Nvidia()

    model = clf.build_model(
        learning_rate=learning_rate,
        dropout_prob=dropout_prob,
        activation=activation,
        use_weights=use_weighs
    )

    # train_perm = np.arange(len(dataset.X_train))
    # np.random.shuffle(train_perm)

    clf.fit(
        batch_generator=dataset.batch_generator,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        colorspace=colorspace
    )

    # This has the unfortunate side-effect of loading all test set images into memory
    # To save on memory, I'd write my own batch generator
    test_score = model.evaluate(
        np.array(list(map(lambda x: preprocess_image(x.center_camera_view(), clf.output_shape), dataset.X_test))),
        dataset.y_test, verbose=1)
    print('Test score:', test_score[0])
    print('Test accuracy:', test_score[1])


# start

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('classifier', 'track1', "The network to train.")
flags.DEFINE_integer('epochs', 2, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")
flags.DEFINE_boolean('use_weights', False, "Whether to use prior trained weights.")
flags.DEFINE_float('lr', 0.001, "Optimizer learning rate.")
flags.DEFINE_float('dropout_prob', 0.5, "Percentage of neurons to misfire during training.")
flags.DEFINE_string('activation', 'elu', "The activation function used by the network.")
flags.DEFINE_string('colorspace', 'yuv', "The colorspace to convert images to during preprocessing phase.")


def main(_):
    train_network(
        classifier=FLAGS.classifier,
        nb_epoch=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.lr,
        dropout_prob=FLAGS.dropout_prob,
        activation=FLAGS.activation,
        use_weighs=FLAGS.use_weights,
        colorspace=FLAGS.colorspace
    )


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
