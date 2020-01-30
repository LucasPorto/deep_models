import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

random.seed(100)
np.random.seed(100)
tf.set_random_seed(100)

from tensorflow.python.keras.losses import mse
from tensorflow.python.keras.metrics import mse as mse_metric
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from data import registration_dataset_mnist
from model import spatial_transformer
from train_utils import WriteOutputImages, dice_coef_loss


def display_results(model : Model, data, n):
    src, tgt = data

    preds = model.predict([src, tgt])
    metrics = model.evaluate(x=[src, tgt], y=tgt)

    samples = np.random.choice(np.arange(preds.shape[0]), n)
    print(samples)
    f, axes = plt.subplots(len(samples), 1)
    for i, s in enumerate(samples):
        axes[i].imshow(np.squeeze(np.hstack([src[i], preds[i], tgt[i]])))

    plt.show()
    print(model.metrics_names)
    print(metrics)


def train():
    train_data, test_data = registration_dataset_mnist(1000, (28, 28), label=3)
    (src_train, tgt_train) = train_data
    (src_test, tgt_test) = test_data

    img_size = src_train.shape[1:3]
    print(img_size)

    conv_filters = [32, 64]

    stn = spatial_transformer(img_size, conv_filters=conv_filters)
    stn.compile(optimizer=Adam(lr=1e-4), loss=mse, metrics=[mse_metric])
    stn.summary()

    mycb = WriteOutputImages(inputs=[src_test[0:10], tgt_test[0:10]], labels=[tgt_test[0:10]])
    stn.fit(x=[src_train, tgt_train], y=tgt_train, validation_split=0.2,
            epochs=500, batch_size=50, verbose=1, callbacks=[mycb])

    return stn, train_data, test_data


if __name__ == '__main__':
    stn, train_data, test_data = train()
    display_results(stn, test_data, 5)