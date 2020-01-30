from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

def registration_dataset_mnist(n, img_shape, label=None):
    (x_train, y_train), _ = mnist.load_data()
    if label:
        sample_indices = np.random.choice(np.where(y_train == label)[0], 1, replace=False)[0]
    else:
        sample_indices = np.random.choice(np.arange(len(x_train)), 1, replace=False)[0]

    target_img = x_train[sample_indices]
    target_img = np.array(Image.fromarray(target_img).resize(img_shape)) / 255.0
    targets = np.array([target_img for _ in range(n)])
    targets = np.expand_dims(targets, axis=-1)

    imgen = ImageDataGenerator(
        rotation_range=90,
        shear_range=50,
        zoom_range=[1.5, 1.7],
        width_shift_range=0.1
    )

    sources = next(imgen.flow(targets, batch_size=n, shuffle=False))

    # Setting aside 20% of data for testing
    sources_train = sources[0:int(n*0.8)]
    targets_train = targets[0:int(n*0.8)]

    sources_test = sources[int(n*0.8):]
    targets_test = targets[int(n*0.8):]

    return (sources_train, targets_train), (sources_test, targets_test)











