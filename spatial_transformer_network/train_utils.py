import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import Callback

import matplotlib.pyplot as plt
import numpy as np
import os


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


class WriteOutputImages(Callback):
    def __init__(self, inputs, labels=None, output_path=None):
        super().__init__()
        self.inputs = inputs

        if labels:
            self.labels = labels
        else:
            self.labels = []

        if output_path:
            self.output_path = output_path
        else:
            self.output_path = os.path.join(os.path.dirname(__file__), 'outputs')

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def on_epoch_end(self, epoch, logs=None):
        get_output = K.function(self.model.inputs, self.model.outputs)
        outputs = get_output(self.inputs)

        blended = [outputs[i]*(0.7) + self.labels[i]*(1-0.7) for i in range(len(outputs))]

        images_list = self.inputs + blended
        f, ax = plt.subplots(len(images_list), 1)

        for a, images in zip(ax, images_list):
            im = a.imshow(np.squeeze(np.hstack(images)))
            f.colorbar(im, ax=a)

        f.savefig(os.path.join(self.output_path, 'out_{:04d}.jpg'.format(epoch)))
        plt.close()
