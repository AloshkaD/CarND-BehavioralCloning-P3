import argparse
import base64
import json

import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import model_from_json

from model import preprocess_image

sio = socketio.Server()
app = Flask(__name__)
model = None

OUTPUT_SHAPE = (40, 80)
MIN_THROTTLE = 0.2
MAX_THROTTLE = 0.5


def get_throttle(steering_angle):
    """
    Simulated throttle modulation

    Simulates a human driving taking their foot off the accellerator
    pedal when approaching a sharp curve.

    The idea here is to find # of standard deviations away from 0 the steering angle is.
    The further away from the mean (i.e., wide turns), the more we lay off the throttle.

    That said, 3 works well in most cases so it's fixed for now.

    :param steering_angle:
    :return:
    """
    stdev = 3
    throttle_modulation = stdev * (-(MAX_THROTTLE - MIN_THROTTLE) * abs(steering_angle))
    # return 0.2
    return throttle_modulation + MAX_THROTTLE


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = preprocess_image(image_array, output_shape=OUTPUT_SHAPE)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = get_throttle(steering_angle)
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    model.summary()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
